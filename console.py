# alerts/console.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


def _now_local() -> str:
    return datetime.now().strftime("%m/%d/%Y %I:%M:%S %p LOCAL")


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str):
            s = x.strip()
            s = s.replace("+", "")
            return float(s)
        return float(x)
    except Exception:
        return None


def _fmt(x: Any, nd: int = 3) -> str:
    v = _to_float(x)
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def _fmt_signed(x: Any, nd: int = 3) -> str:
    v = _to_float(x)
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}"


class ConsoleAlerter:
    def print_signal(self, out: Dict[str, Any]) -> None:
        if not out:
            return

        symbol = str(out.get("symbol", ""))
        side = str(out.get("side", "")).upper()
        tier = str(out.get("tier", "")).strip()
        origin = str(out.get("origin", out.get("BOT_INSTANCE_NAME", "UNKNOWN"))).strip()

        ofi = out.get("ofi")
        vpin = out.get("vpin")
        sigma_next = out.get("sigma_next")
        lev_cap = out.get("lev_cap")
        spread_bps = out.get("spread_bps")
        micro_ticks = out.get("microprice_ticks")

        entry = out.get("entry")
        stop = out.get("stop")

        tp = out.get("tp")
        if tp is None:
            tp = out.get("tp2")

        vf_side = out.get("vf_side")
        vf_h = out.get("vf_horizon_sec")

        kyle_lambda = out.get("kyle_lambda")

        print("=" * 95)
        if tier:
            print(f"[{_now_local()}] [{origin}] [SIGNAL] {symbol} ({tier})  {side}")
        else:
            print(f"[{_now_local()}] [{origin}] [SIGNAL] {symbol}  {side}")

        if ofi is not None and vpin is not None:
            print(f"  ofi={_fmt_signed(ofi,3)}  vpin={_fmt(vpin,3)}")

        if sigma_next is not None and lev_cap is not None:
            print(f"  sigma_next={_fmt(sigma_next,6)}  lev_cap~{_fmt(lev_cap,2)}x")

        if spread_bps is not None and micro_ticks is not None:
            print(f"  spread={_fmt(spread_bps,2)} bps  microprice_ticks={_fmt_signed(micro_ticks,2)}")

        if entry is not None:
            print(f"  PRIMARY ENTRY: {_fmt(entry,4)}")
        if stop is not None:
            print(f"  STOP: {_fmt(stop,4)}")

        if tp is not None:
            print(f"  TP: {_fmt(tp,4)}")

        if vf_side is not None and vf_h is not None:
            try:
                h = int(vf_h)
            except Exception:
                h = vf_h
            print(f"  🔔 VF: {str(vf_side).upper()} volume likely within ~{h}s")

        if kyle_lambda is not None:
            print(f"  KL: {_fmt(kyle_lambda,1)}")

        print("=" * 95)
