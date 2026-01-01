from __future__ import annotations

import math
import time
from datetime import datetime

from alerts.console import ConsoleAlerter
from alerts.telegram import TelegramAlerter
from config.settings import Settings
from engine.decision_engine import DecisionEngine
from engine.tier_manager import TierManager, TierMetrics
from exchange.binance_public import BinanceFuturesPublicClient
from exchange.binance_ws import BinanceFuturesWSHub


def now_local() -> str:
    return datetime.now().strftime("%m/%d/%Y %I:%M:%S %p LOCAL")


def _tier_bucket_base(cfg: Settings, sym: str, tier_label: str) -> float:
    if sym in ("BTC_USDT", "ETH_USDT") or tier_label == "T1":
        return float(getattr(cfg, "VPIN_BUCKET_T1_USD", 200_000))
    if tier_label == "T2":
        return float(getattr(cfg, "VPIN_BUCKET_T2_USD", 90_000))
    return float(getattr(cfg, "VPIN_BUCKET_T3_USD", 45_000))


def _bucket_scale_from_volume(cfg: Settings, vol_24h: float) -> float:
    ref = float(getattr(cfg, "VPIN_BUCKET_VOL_REF_USD", 25_000_000))
    if ref <= 0:
        return 1.0
    x = max(vol_24h, 0.0) / ref
    scale = math.sqrt(max(x, 1e-9))
    smin = float(getattr(cfg, "VPIN_BUCKET_SCALE_MIN", 0.50))
    smax = float(getattr(cfg, "VPIN_BUCKET_SCALE_MAX", 3.00))
    return max(smin, min(smax, scale))


def calc_vpin_bucket(cfg: Settings, sym: str, tier_label: str, vol_24h: float) -> float:
    base = _tier_bucket_base(cfg, sym, tier_label)
    return float(base * _bucket_scale_from_volume(cfg, vol_24h))


def build_watchlist_snapshot(cfg: Settings, binance: BinanceFuturesPublicClient) -> dict[str, TierMetrics]:
    valid = set(binance.list_usdt_perp_symbols())
    tickers = binance.tickers_24h()

    rows: list[tuple[str, float]] = []
    for t in tickers:
        sym = t.get("symbol")
        if not sym or sym not in valid:
            continue
        try:
            qv = float(t.get("quoteVolume", 0.0))
        except Exception:
            continue
        rows.append((sym, qv))

    rows.sort(key=lambda x: x[1], reverse=True)
    top = rows[: cfg.WATCHLIST_TOP_N]

    snapshot: dict[str, TierMetrics] = {}
    for sym, qv in top:
        sym_u = sym.replace("USDT", "_USDT")
        snapshot[sym_u] = TierMetrics(volume_24h=qv, spread_bps=0.0, vpin=None, sigma=None)

    return snapshot


def main():
    print("[start] inst-signal-bot starting (WS mode)...")
    cfg = Settings()

    binance = BinanceFuturesPublicClient()
    engine = DecisionEngine(cfg)
    tier = TierManager(cfg)
    console = ConsoleAlerter()

    tg: TelegramAlerter | None = None
    if getattr(cfg, "TELEGRAM_ENABLED", False):
        token = str(getattr(cfg, "TELEGRAM_BOT_TOKEN", "")).strip()
        chat_id = str(getattr(cfg, "TELEGRAM_CHAT_ID", "")).strip()
        debug = bool(getattr(cfg, "TELEGRAM_DEBUG", False))
        if token and chat_id:
            tg = TelegramAlerter(
                token=token,
                chat_id=chat_id,
                debug=debug,
                instance_name=str(getattr(cfg, "BOT_INSTANCE_NAME", "UNKNOWN")),
            )

    # ----------------------------
    # Websocket hub
    # ----------------------------
    ws = BinanceFuturesWSHub(symbols=list(cfg.SYMBOLS), depth_levels=int(cfg.DEPTH_LEVELS))
    ws.start()

    # ----------------------------
    # Bootstrap Tier-1
    # ----------------------------
    print("[boot] starting bootstrap loop")
    for sym in cfg.SYMBOLS:
        bucket = calc_vpin_bucket(cfg, sym, "T1", vol_24h=float(getattr(cfg, "VPIN_BUCKET_VOL_REF_USD", 25_000_000)))
        engine.bootstrap_symbol(None, binance, sym, vpin_bucket_notional_usd=bucket, tier="T1")
        print(f"[boot] {sym} initialized (Tier-1) vpin_bucket_usd={bucket:,.0f}")

    last_snapshot: dict[str, TierMetrics] = {}
    last_snapshot_ts = 0.0
    last_confirm_ts = 0.0
    last_hb = time.time()

    last_any_api_ts = time.time()
    last_depth_ts = time.time()
    last_trades_ts = time.time()

    last_book_ts: dict[str, float] = {s: 0.0 for s in cfg.SYMBOLS}
    last_trade_ts: dict[str, float] = {s: 0.0 for s in cfg.SYMBOLS}

    while True:
        loop_t0 = time.time()
        now = loop_t0

        # Tier refresh
        if (now - last_snapshot_ts) >= cfg.TIER_REFRESH_SEC or not last_snapshot:
            try:
                snap = build_watchlist_snapshot(cfg, binance)
                tier.refresh_universe(snap)
                last_snapshot = snap
                last_snapshot_ts = now

                active_syms = sorted(list(tier.tier1 | tier.tier2 | tier.t2_candidates))
                ws.update_symbols(active_syms)

                for sym in list(tier.t2_candidates):
                    if sym not in engine.state:
                        tm = last_snapshot.get(sym)
                        vol_24h = float(tm.volume_24h) if tm else 0.0
                        bucket = calc_vpin_bucket(cfg, sym, "CAND", vol_24h=vol_24h)
                        engine.bootstrap_symbol(None, binance, sym, vpin_bucket_notional_usd=bucket, tier="CAND")
                        last_book_ts[sym] = 0.0
                        last_trade_ts[sym] = 0.0

                print(f"[tier] refresh complete | T1={len(tier.tier1)} T2={len(tier.tier2)} CAND={len(tier.t2_candidates)}")
            except Exception as e:
                print(f"[tier-warn] refresh failed: {e}")

        # Candidate confirm cadence
        if (now - last_confirm_ts) >= cfg.T2_CONFIRM_SEC:
            live: dict[str, TierMetrics] = {}
            for sym in tier.t2_candidates:
                st = engine.state.get(sym)
                if not st:
                    continue
                snap_m = last_snapshot.get(sym)
                vol_24h = float(snap_m.volume_24h) if snap_m else 0.0
                live[sym] = TierMetrics(
                    volume_24h=vol_24h,
                    spread_bps=0.0,
                    vpin=float(st.vpin.value()),
                    sigma=float(st.garch.sigma()),
                )
            tier.confirm_candidates(live)
            last_confirm_ts = now

            for sym in list(tier.tier2):
                st = engine.state.get(sym)
                if st and getattr(st, "tier", "") != "T2":
                    st.tier = "T2"

        active = sorted(list(tier.tier1 | tier.tier2 | tier.t2_candidates))

        any_signal = False
        for sym in active:
            st = engine.state.get(sym)
            if not st:
                continue

            book = ws.get_latest_book(sym)
            if book is not None:
                last_any_api_ts = now
                last_depth_ts = now
                last_book_ts[sym] = now

                trades = ws.pop_trades(sym, max_n=250)
                if trades:
                    engine.update_from_deals(sym, trades)
                    last_any_api_ts = now
                    last_trades_ts = now
                    last_trade_ts[sym] = now

                out = engine.on_book(sym, book["bids"], book["asks"])
                if not out:
                    continue

                if out.get("type", "ENTRY") == "ENTRY" and str(out.get("side", "")).upper() in ("LONG", "SHORT"):
                    any_signal = True
                    if sym in tier.tier1:
                        out["tier"] = "T1"
                    elif sym in tier.tier2:
                        out["tier"] = "T2"
                    else:
                        out["tier"] = "CAND"

                    if sym in tier.tier2:
                        tier.record_side(sym, out.get("side", ""))

                    hint = None
                    vf = getattr(engine, "vf", None)
                    if vf is not None and getattr(cfg, "VOLUME_FORECAST_ENABLED", False):
                        try:
                            hint = vf.get_hint_for_signal(sym, out["side"])
                        except Exception:
                            hint = None
                    if hint is not None:
                        vside, horizon = hint
                        out["vf_side"] = vside
                        out["vf_horizon_sec"] = int(horizon)

                    console.print_signal(out)
                    if tg:
                        tg.send_entry(out)

        # Heartbeat
        if (now - last_hb) >= cfg.HEARTBEAT_SEC:
            status = "signal" if any_signal else "no_signal"

            api_age = max(0.0, time.time() - last_any_api_ts)
            depth_age = max(0.0, time.time() - last_depth_ts)
            trades_age = max(0.0, time.time() - last_trades_ts)

            ws_lastmsg_age = max(0.0, time.time() - float(ws.stats.last_msg_ts))
            ws_depth = int(ws.stats.depth_msgs)
            ws_trades = int(ws.stats.trade_msgs)
            ws_reconnects = int(ws.stats.reconnects)
            ws_err = str(ws.stats.last_err or "")

            trace_top = engine.trace.most_common(6) if hasattr(engine, "trace") else []

            def _top_stale(d: dict[str, float], n: int = 2):
                rows = []
                for k, ts in d.items():
                    if ts <= 0:
                        rows.append((k, float("inf")))
                    else:
                        rows.append((k, max(0.0, now - ts)))
                rows.sort(key=lambda x: x[1], reverse=True)
                return rows[:n]

            stale_books_top = _top_stale(last_book_ts, 2)
            stale_trades_top = _top_stale(last_trade_ts, 2)

            vpin_ready_n = 0
            vpin_pass_n = 0
            vpin_sum = 0.0
            vpin_cnt = 0
            setups = 0

            for s in active:
                st = engine.state.get(s)
                if not st:
                    continue
                if getattr(st, "setup", None) is not None:
                    setups += 1
                try:
                    v = float(st.vpin.value())
                    br = int(st.vpin.buckets_ready())
                    vpin_sum += v
                    vpin_cnt += 1
                    if br >= int(getattr(cfg, "VPIN_MIN_BUCKETS_READY", 2)):
                        vpin_ready_n += 1
                        if v >= float(cfg.VPIN_MIN):
                            vpin_pass_n += 1
                except Exception:
                    pass

            avg_vpin = (vpin_sum / vpin_cnt) if vpin_cnt else 0.0

            print(
                f"[hb] {now_local()} | T1={len(tier.tier1)} T2={len(tier.tier2)} CAND={len(tier.t2_candidates)} alive | {status} | "
                f"api_age={api_age:.1f}s depth_age={depth_age:.1f}s trades_age={trades_age:.1f}s | "
                f"ws_depth={ws_depth} ws_trades={ws_trades} ws_lastmsg_age={ws_lastmsg_age:.1f}s ws_reconnects={ws_reconnects} ws_err={ws_err} | "
                f"diag: setups={setups} vpin_ready={vpin_ready_n}/{max(len(active),1)} vpin_pass={vpin_pass_n} avg_vpin={avg_vpin:.3f} VPIN_MIN={cfg.VPIN_MIN:.2f} "
                f"VOL_KILL={cfg.VOL_KILL:.3f} KL={'ON' if getattr(cfg,'KYLE_LAMBDA_ENABLED',False) else 'OFF'} | "
                f"trace_top={trace_top} stale_books_top={stale_books_top} stale_trades_top={stale_trades_top}"
            )
            last_hb = now

        time.sleep(cfg.LOOP_SLEEP_SEC)


if __name__ == "__main__":
    main()
