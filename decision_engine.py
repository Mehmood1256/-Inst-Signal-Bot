from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque, Counter

import numpy as np

from config.settings import Settings
from indicators.returns import log_return
from indicators.garch import Garch11Online
from models.microstructure import mid_price, order_flow_imbalance, microprice_ticks, spread_bps
from models.vpin import VPIN
from models.calibration import AdaptiveThresholds
from models.jump_risk import JumpRiskModel
from models.kyle_lambda import KyleLambda
from models.volume_forecast import VolumeForecaster


def _round_to_tick(px: float, tick: float) -> float:
    if tick <= 0:
        return float(px)
    return round(px / tick) * tick


def _bps_move(a: float, b: float) -> float:
    if a <= 0:
        return 0.0
    return (b - a) / a * 10_000.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class _EntrySetup:
    side: str
    created_ts: float
    entry_anchor: float
    ofi0: float
    micro0: float
    vpin0: float
    sigma0: float
    spread0: float
    confirm_hits: int = 0


@dataclass
class _SymState:
    tick: float
    last_mid: float
    garch: Garch11Online
    vpin: VPIN
    adapt: Optional[AdaptiveThresholds]
    jump: Optional[JumpRiskModel]

    tier: str  # "T1" | "T2" | "CAND"

    regime: str
    last_regime_change_ts: float
    pending_dir: str
    pending_count: int

    setup: Optional[_EntrySetup]
    last_trade_id: Optional[int]

    kyle: Optional[KyleLambda]

    mid_hist: Deque[Tuple[float, float]]
    spread_ewma: float
    spread_samples: int

    last_signal_ts: float
    last_signal_side: str
    last_signal_mid: float


class DecisionEngine:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.state: Dict[str, _SymState] = {}

        # tracer (must stay)
        self.trace = Counter()  # type: ignore[var-annotated]

        # Volume forecaster
        self.vf: Optional[VolumeForecaster] = None
        if getattr(cfg, "VOLUME_FORECAST_ENABLED", False):
            self.vf = VolumeForecaster(
                hawkes_tau_sec=float(getattr(cfg, "VF_HAWKES_TAU_SEC", 30.0)),
                hawkes_z_window=int(getattr(cfg, "VF_HAWKES_Z_WINDOW", 240)),
                l2_band_bps=float(getattr(cfg, "VF_L2_BAND_BPS", 10.0)),
                l2_ema_alpha=float(getattr(cfg, "VF_L2_EMA_ALPHA", 0.03)),
                burst_z_thr=float(getattr(cfg, "VF_BURST_Z_THR", 1.5)),
                p_dir_thr=float(getattr(cfg, "VF_P_DIR_THR", 0.65)),
                combine_thr=float(getattr(cfg, "VF_COMBINE_THR", 0.63)),
                cooldown_sec=int(getattr(cfg, "VF_COOLDOWN_SEC", 60)),
                w_hawkes=float(getattr(cfg, "VF_W_HAWKES", 0.55)),
                w_l2=float(getattr(cfg, "VF_W_L2", 0.45)),
            )

    # -------------------------
    # tier-aware thresholds
    # -------------------------
    def _tier_params(self, tier: str) -> Tuple[float, float, float, float]:
        """
        Returns (ofi_thr, micro_thr, vpin_min, pullback_min_bps)
        T1/T2 loosened, CAND uses base cfg.

        NOTE: pullback is disabled globally in on_book (final fix),
        pb value is retained only for compatibility / tracing.
        """
        cfg = self.cfg

        base_ofi = float(cfg.OFI_THR)
        base_micro = float(cfg.MICRO_TICKS_MIN)
        base_vpin = float(cfg.VPIN_MIN)
        base_pb = float(getattr(cfg, "PULLBACK_MIN_BPS", 0.0))

        if tier == "T1":
            ofi = base_ofi * float(getattr(cfg, "T1_OFI_MULT", 1.0))
            micro = base_micro * float(getattr(cfg, "T1_MICRO_MULT", 1.0))
            vpin = float(getattr(cfg, "T1_VPIN_MIN", base_vpin))
            pb = float(getattr(cfg, "T1_PULLBACK_MIN_BPS", base_pb))
            return ofi, micro, vpin, pb

        if tier == "T2":
            ofi = base_ofi * float(getattr(cfg, "T2_OFI_MULT", 1.0))
            micro = base_micro * float(getattr(cfg, "T2_MICRO_MULT", 1.0))
            vpin = float(getattr(cfg, "T2_VPIN_MIN", base_vpin))
            pb = float(getattr(cfg, "T2_PULLBACK_MIN_BPS", base_pb))
            return ofi, micro, vpin, pb

        return base_ofi, base_micro, base_vpin, base_pb

    # -------------------------
    # bootstrap
    # -------------------------
    def bootstrap_symbol(
        self,
        mexc_client: Any,
        binance_client: Any,
        symbol: str,
        vpin_bucket_notional_usd: Optional[float] = None,
        tier: str = "CAND",
    ) -> Tuple[float, float]:
        info = binance_client.contract_detail(symbol)
        tick = float(info["priceUnit"])

        kl = binance_client.kline(symbol, interval=self.cfg.KLINE_INTERVAL, limit=self.cfg.KLINES_LIMIT)
        closes = [float(x[4]) for x in kl]
        if len(closes) < 5:
            raise RuntimeError("not enough klines")

        rets = [log_return(closes[i - 1], closes[i]) for i in range(1, len(closes))]
        longrun_var = float(np.var(rets)) if rets else 1e-8
        longrun_var = max(longrun_var, 1e-12)

        garch = Garch11Online.from_longrun_var(
            longrun_var=longrun_var,
            alpha=self.cfg.GARCH_ALPHA,
            beta=self.cfg.GARCH_BETA,
        )
        for r in rets[-min(200, len(rets)) :]:
            garch.update(float(r))

        bucket = float(vpin_bucket_notional_usd) if vpin_bucket_notional_usd is not None else 75_000.0
        bucket = max(bucket, 1_000.0)

        vpin = VPIN(bucket_notional_usd=bucket, lookback_buckets=int(self.cfg.VPIN_LOOKBACK_BUCKETS))

        adapt = None
        if self.cfg.ADAPTIVE_ENABLED:
            adapt = AdaptiveThresholds(
                window=self.cfg.ADAPT_WINDOW_SAMPLES,
                min_samples=self.cfg.ADAPT_MIN_SAMPLES,
                refresh_sec=self.cfg.ADAPT_REFRESH_SEC,
                ofi_q=self.cfg.OFI_Q,
                micro_q=self.cfg.MICRO_Q,
                vpin_q=self.cfg.VPIN_Q,
                ofi_min=self.cfg.OFI_THR_MIN,
                ofi_max=self.cfg.OFI_THR_MAX,
                micro_min=self.cfg.MICRO_TICKS_MIN_MIN,
                micro_max=self.cfg.MICRO_TICKS_MIN_MAX,
                vpin_min=self.cfg.VPIN_MIN_MIN,
                vpin_max=self.cfg.VPIN_MIN_MAX,
            )

        jump = None
        if self.cfg.JUMP_RISK_ENABLED:
            jump = JumpRiskModel(
                window=self.cfg.JUMP_Z_WINDOW,
                w_vpin=self.cfg.JUMP_W_VPIN,
                w_spread=self.cfg.JUMP_W_SPREAD,
                w_abs_ofi=self.cfg.JUMP_W_ABS_OFI,
                w_abs_micro=self.cfg.JUMP_W_ABS_MICRO,
            )

        kyle = None
        if getattr(self.cfg, "KYLE_LAMBDA_ENABLED", False):
            kyle = KyleLambda(
                window_sec=int(getattr(self.cfg, "KL_WINDOW_SEC", 60)),
                ema_alpha=float(getattr(self.cfg, "KL_EMA_ALPHA", 0.20)),
                min_notional_usd=float(getattr(self.cfg, "KL_MIN_NOTIONAL_USD", 25_000.0)),
            )

        last_mid = float(closes[-1])
        now = time.time()

        mh: Deque[Tuple[float, float]] = deque(maxlen=3000)
        mh.append((now, last_mid))

        self.state[symbol] = _SymState(
            tick=tick,
            last_mid=last_mid,
            garch=garch,
            vpin=vpin,
            adapt=adapt,
            jump=jump,
            tier=tier,
            regime="NONE",
            last_regime_change_ts=0.0,
            pending_dir="NONE",
            pending_count=0,
            setup=None,
            last_trade_id=None,
            kyle=kyle,
            mid_hist=mh,
            spread_ewma=0.0,
            spread_samples=0,
            last_signal_ts=0.0,
            last_signal_side="NONE",
            last_signal_mid=last_mid,
        )

        if kyle is not None:
            kyle.update_mid(last_mid, ts=now)

        return tick, float(np.sqrt(longrun_var))

    # -------------------------
    # trades
    # -------------------------
    def _filter_new_trades(self, st: _SymState, trades: List[dict]) -> List[dict]:
        if not trades:
            return []
        if "id" not in trades[0]:
            return trades

        def _tid(t: dict) -> int:
            try:
                return int(t.get("id", 0))
            except Exception:
                return 0

        trades_sorted = sorted(trades, key=_tid)
        last_id = st.last_trade_id
        if last_id is None:
            st.last_trade_id = _tid(trades_sorted[-1])
            return trades_sorted

        new_trades = [t for t in trades_sorted if _tid(t) > last_id]
        if new_trades:
            st.last_trade_id = _tid(new_trades[-1])
        return new_trades

    def update_from_deals(self, symbol: str, trades: List[dict]) -> None:
        st = self.state.get(symbol)
        if not st:
            return

        new_trades = self._filter_new_trades(st, trades)
        if not new_trades:
            return

        st.vpin.update(new_trades)

        if self.vf is not None:
            self.vf.update_trades(symbol, new_trades)

        if st.kyle is not None:
            signed_sum = 0.0
            abs_sum = 0.0
            for t in new_trades:
                is_buyer_maker = bool(t.get("isBuyerMaker", False))
                try:
                    price = float(t.get("price", 0.0))
                    qty = float(t.get("qty", 0.0))
                except Exception:
                    price, qty = 0.0, 0.0

                quote_qty = t.get("quoteQty", None)
                try:
                    notional = float(quote_qty) if quote_qty is not None else price * qty
                except Exception:
                    notional = price * qty

                notional = max(0.0, float(notional))
                if notional <= 0:
                    continue
                abs_sum += notional
                signed_sum += (-notional if is_buyer_maker else +notional)

            if abs_sum > 0:
                st.kyle.update_trades(signed_usd=signed_sum, abs_usd=abs_sum, ts=time.time())

    # -------------------------
    # helpers
    # -------------------------
    def _mid_move_bps_over(self, st: _SymState, now: float, window_sec: int) -> float:
        if not st.mid_hist:
            return 0.0
        t0 = now - float(window_sec)
        old = None
        latest = st.mid_hist[-1][1]
        for ts, mid in st.mid_hist:
            if ts <= t0:
                old = mid
            else:
                break
        if old is None:
            return 0.0
        return _bps_move(old, latest)

    def _dynamic_spread_limit(self, st: _SymState) -> float:
        cfg = self.cfg
        hard = float(cfg.REL_SPREAD_MAX_BPS)

        if not bool(getattr(cfg, "SPREAD_EWMA_ENABLED", True)):
            return hard
        if st.spread_samples < int(getattr(cfg, "SPREAD_EWMA_MIN_SAMPLES", 60)):
            return hard

        mult = float(getattr(cfg, "SPREAD_EWMA_MULT", 1.7))
        lim = st.spread_ewma * mult if st.spread_ewma > 0 else hard
        return min(hard, float(lim))

    def _kyle_gate_ok(self, symbol: str, side: str, now: float) -> Tuple[bool, Optional[Dict[str, float]]]:
        st = self.state[symbol]
        k = st.kyle
        if k is None:
            return True, None

        snap = k.snapshot()
        if snap is None:
            return False, None

        lam_min = float(getattr(self.cfg, "KL_LAMBDA_MIN_BPS_PER_MUSD", 0.0))
        flow_frac = float(getattr(self.cfg, "KL_FLOW_SIGN_MIN_FRAC", 0.0))
        min_notional = float(getattr(self.cfg, "KL_MIN_NOTIONAL_USD", 0.0))
        min_mid_move = float(getattr(self.cfg, "KL_MIN_MID_MOVE_BPS", 0.0))
        win = int(getattr(self.cfg, "KL_WINDOW_SEC", 60))

        notional = float(snap.notional_usd)
        signed = float(snap.flow_signed_usd)
        lam = float(snap.lambda_bps_per_musd)

        mid_move = self._mid_move_bps_over(st, now=now, window_sec=win)

        meta = {"kyle_lambda": lam}

        if notional < min_notional:
            return False, meta
        if lam < lam_min:
            return False, meta
        if abs(mid_move) < min_mid_move:
            return False, meta

        if side == "LONG" and mid_move <= 0:
            return False, meta
        if side == "SHORT" and mid_move >= 0:
            return False, meta

        need = flow_frac * notional
        if side == "LONG":
            if signed < need:
                return False, meta
        else:
            if signed > -need:
                return False, meta

        return True, meta

    def _signal_allowed(self, st: _SymState, side: str, mid: float, now: float) -> bool:
        cfg = self.cfg

        cd = float(getattr(cfg, "SIGNAL_COOLDOWN_SEC", 0))
        if cd > 0 and (now - float(st.last_signal_ts)) < cd:
            self.trace["emit:cooldown"] += 1
            return False

        flip_cd = float(getattr(cfg, "SIGNAL_FLIP_COOLDOWN_SEC", 0))
        if (
            flip_cd > 0
            and st.last_signal_side in ("LONG", "SHORT")
            and side in ("LONG", "SHORT")
            and st.last_signal_side != side
            and (now - float(st.last_signal_ts)) < flip_cd
        ):
            self.trace["emit:flip_cooldown"] += 1
            return False

        min_move = float(getattr(cfg, "SIGNAL_MIN_MOVE_BPS_TO_RESIGNAL", 0))
        if min_move > 0 and abs(_bps_move(float(st.last_signal_mid), float(mid))) < min_move:
            self.trace["emit:min_move"] += 1
            return False

        return True

    def _build_levels(
        self,
        symbol: str,
        side: str,
        mid: float,
        sigma_next: float,
        ofi: float,
        vpin_val: float,
        spr_bps: float,
        micro_ticks: float,
        kyle_meta: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        st = self.state[symbol]

        entry = mid
        lev_cap = min(cfg.L_MAX, max(1.0, cfg.TARGET_RISK_FRAC / max(sigma_next, 1e-9)))

        stop_dist = cfg.K_SL * sigma_next * mid
        stop_dist = max(stop_dist, st.tick * 2.0)

        stop = entry - stop_dist if side == "LONG" else entry + stop_dist

        entry = _round_to_tick(entry, st.tick)
        stop = _round_to_tick(stop, st.tick)

        R = max(abs(entry - stop), st.tick * 2.0)

        tp2 = entry + cfg.TP2_R * R if side == "LONG" else entry - cfg.TP2_R * R
        tp2 = _round_to_tick(tp2, st.tick)

        out: Dict[str, Any] = {
            "ts": time.time(),
            "type": "ENTRY",
            "symbol": symbol,
            "side": side,
            "ofi": float(ofi),
            "vpin": float(vpin_val),
            "sigma_next": float(sigma_next),
            "lev_cap": float(lev_cap),
            "spread_bps": float(spr_bps),
            "microprice_ticks": float(micro_ticks),
            "entry": float(entry),
            "stop": float(stop),
            "tp2": float(tp2),
            "origin": str(getattr(cfg, "BOT_INSTANCE_NAME", "UNKNOWN")),
        }
        if kyle_meta:
            out.update(kyle_meta)
        return out

    # -------------------------
    # main: book processing
    # -------------------------
    def on_book(self, symbol: str, bids: List[List[str]], asks: List[List[str]]) -> Optional[Dict[str, Any]]:
        st = self.state.get(symbol)
        if not st:
            return None

        cfg = self.cfg
        now = time.time()

        mid = mid_price(bids, asks)
        if mid <= 0:
            self.trace["core:none"] += 1
            return None

        if self.vf is not None:
            try:
                self.vf.update_book(symbol, bids, asks)
            except Exception:
                pass

        st.mid_hist.append((now, float(mid)))
        max_win = max(int(getattr(cfg, "PULLBACK_WINDOW_SEC", 60)), int(getattr(cfg, "KL_WINDOW_SEC", 60)))
        cutoff = now - float(max_win) - 10.0
        while st.mid_hist and st.mid_hist[0][0] < cutoff:
            st.mid_hist.popleft()

        if st.kyle is not None:
            st.kyle.update_mid(mid, ts=now)

        prev_mid = float(st.last_mid)
        r = log_return(prev_mid, mid)
        st.last_mid = mid

        st.garch.update(r)
        sigma_next = float(st.garch.sigma())

        if sigma_next >= float(cfg.VOL_KILL):
            self.trace["gate:vol_kill"] += 1
            st.setup = None
            return None

        ofi = float(order_flow_imbalance(bids, asks, k=cfg.BOOK_LEVELS_K))
        micro_ticks = float(microprice_ticks(bids, asks, tick=st.tick))
        spr_bps = float(spread_bps(bids, asks))

        if bool(getattr(cfg, "SPREAD_EWMA_ENABLED", True)):
            a = float(getattr(cfg, "SPREAD_EWMA_ALPHA", 0.05))
            if st.spread_samples <= 0:
                st.spread_ewma = spr_bps
            else:
                st.spread_ewma = (1.0 - a) * float(st.spread_ewma) + a * spr_bps
            st.spread_samples += 1

        spread_limit = self._dynamic_spread_limit(st)
        if spr_bps > spread_limit:
            self.trace["gate:spread"] += 1
            st.setup = None
            return None

        vol_floor = float(getattr(cfg, "VOL_FLOOR", 0.0))
        if vol_floor > 0 and sigma_next < vol_floor:
            if (abs(ofi) < float(cfg.OFI_THR) * float(getattr(cfg, "VOL_FLOOR_OFI_MULT", 1.35))) or (
                abs(micro_ticks) < float(cfg.MICRO_TICKS_MIN) * float(getattr(cfg, "VOL_FLOOR_MICRO_MULT", 1.35))
            ):
                self.trace["gate:vol_floor"] += 1
                st.setup = None
                return None

        vpin_val = float(st.vpin.value())
        vpin_ready = st.vpin.buckets_ready() >= int(getattr(cfg, "VPIN_MIN_BUCKETS_READY", 2))

        if bool(getattr(cfg, "VPIN_HARD_WARMUP_BLOCK", True)) and not vpin_ready:
            self.trace["gate:vpin_warmup"] += 1
            st.setup = None
            return None

        # tier thresholds
        tier = getattr(st, "tier", "CAND")
        ofi_thr, micro_thr, vpin_thr, _pb_thr = self._tier_params(tier)

        if st.adapt is not None:
            st.adapt.update(ofi, micro_ticks, vpin_val)
            ofi_thr, micro_thr, _ = st.adapt.thresholds_or(ofi_thr, micro_thr, vpin_thr)

        # ============================================================
        # FINAL FIX: Pullback is DISABLED for ALL tiers
        # ============================================================
        pull_ok_long = True
        pull_ok_short = True
        # (do not increment core:fail_pullback anymore — it will naturally drop out)

        # setup confirmation -> emit
        if cfg.ENTRY_CONFIRM_ENABLED and st.setup is not None:
            setup = st.setup

            if (now - setup.created_ts) > float(cfg.ENTRY_CONFIRM_WINDOW_SEC):
                self.trace["confirm:timeout"] += 1
                st.setup = None
                return None

            chase_bps = abs(_bps_move(setup.entry_anchor, mid))
            max_chase = float(getattr(cfg, "ENTRY_MAX_CHASE_BPS_SPR_MULT", 3.0)) * float(spr_bps)
            max_chase = _clamp(
                max_chase,
                float(getattr(cfg, "ENTRY_MAX_CHASE_BPS_MIN", 6.0)),
                float(getattr(cfg, "ENTRY_MAX_CHASE_BPS_MAX", 12.0)),
            )
            if chase_bps > max_chase:
                self.trace["confirm:chase"] += 1
                st.setup = None
                return None

            spr_mult = float(getattr(cfg, "SPREAD_CONFIRM_MULT", 1.15))
            if spr_bps > float(setup.spread0) * spr_mult:
                self.trace["confirm:spread_worse"] += 1
                st.setup = None
                return None

            if st.jump is not None:
                jr = st.jump.update(vpin=vpin_val, spread_bps=spr_bps, ofi=ofi, micro_ticks=micro_ticks)
                if jr.score >= float(cfg.JUMP_SCORE_THR):
                    self.trace["confirm:jump"] += 1
                    st.setup = None
                    return None

            if setup.side == "LONG":
                ok_dir = (ofi > 0) and (micro_ticks > 0) and (vpin_val >= float(vpin_thr)) and pull_ok_long
                ok_strength = (ofi >= float(cfg.ENTRY_CONFIRM_OFI_FRAC) * float(setup.ofi0)) and (
                    micro_ticks >= float(cfg.ENTRY_CONFIRM_MICRO_FRAC) * float(setup.micro0)
                )
            else:
                ok_dir = (ofi < 0) and (micro_ticks < 0) and (vpin_val >= float(vpin_thr)) and pull_ok_short
                ok_strength = (ofi <= float(cfg.ENTRY_CONFIRM_OFI_FRAC) * float(setup.ofi0)) and (
                    micro_ticks <= float(cfg.ENTRY_CONFIRM_MICRO_FRAC) * float(setup.micro0)
                )

            kyle_ok, kyle_meta = True, None
            if getattr(cfg, "KYLE_LAMBDA_ENABLED", False):
                kyle_ok, kyle_meta = self._kyle_gate_ok(symbol=symbol, side=setup.side, now=now)
                if not kyle_ok:
                    self.trace["confirm:kyle_fail"] += 1

            if ok_dir and ok_strength and kyle_ok:
                setup.confirm_hits += 1
            else:
                setup.confirm_hits = max(setup.confirm_hits - 1, 0)

            if setup.confirm_hits >= int(cfg.ENTRY_CONFIRM_MIN_BOOKS):
                if not self._signal_allowed(st, side=setup.side, mid=mid, now=now):
                    st.setup = None
                    return None

                out = self._build_levels(
                    symbol=symbol,
                    side=setup.side,
                    mid=mid,
                    sigma_next=sigma_next,
                    ofi=ofi,
                    vpin_val=vpin_val,
                    spr_bps=spr_bps,
                    micro_ticks=micro_ticks,
                    kyle_meta=kyle_meta,
                )

                st.last_signal_ts = now
                st.last_signal_side = setup.side
                st.last_signal_mid = float(mid)

                st.setup = None
                self.trace["emit:entry"] += 1
                return out

            return None

        # core desire
        want = "NONE"
        if (ofi > ofi_thr) and (micro_ticks > micro_thr) and (vpin_val >= float(vpin_thr)) and pull_ok_long:
            want = "LONG"
        elif (ofi < -ofi_thr) and (micro_ticks < -micro_thr) and (vpin_val >= float(vpin_thr)) and pull_ok_short:
            want = "SHORT"

        if want == "NONE":
            self.trace["core:none"] += 1
            return None

        # per-failure tracers
        if abs(ofi) < ofi_thr:
            self.trace["core:fail_ofi"] += 1
        if abs(micro_ticks) < micro_thr:
            self.trace["core:fail_micro"] += 1
        if vpin_val < float(vpin_thr):
            self.trace["core:fail_vpin"] += 1

        if st.regime != "NONE" and (now - float(st.last_regime_change_ts)) < float(cfg.REGIME_LOCK_SEC):
            self.trace["gate:regime_lock"] += 1
            return None

        if want != st.pending_dir:
            st.pending_dir = want
            st.pending_count = 1
            self.trace["core:pending_reset"] += 1
            return None

        st.pending_count += 1
        required = int(cfg.ENTER_PERSIST_N) if st.regime == "NONE" else int(cfg.FLIP_PERSIST_N)
        if st.pending_count < required:
            self.trace["core:persist_wait"] += 1
            return None

        if want == st.regime:
            self.trace["core:same_regime"] += 1
            return None

        if not self._signal_allowed(st, side=want, mid=mid, now=now):
            return None

        st.regime = want
        st.last_regime_change_ts = now
        st.pending_dir = "NONE"
        st.pending_count = 0

        st.setup = _EntrySetup(
            side=want,
            created_ts=now,
            entry_anchor=mid,
            ofi0=float(ofi),
            micro0=float(micro_ticks),
            vpin0=float(vpin_val),
            sigma0=float(sigma_next),
            spread0=float(spr_bps),
            confirm_hits=0,
        )
        self.trace["setup:created"] += 1
        return None
