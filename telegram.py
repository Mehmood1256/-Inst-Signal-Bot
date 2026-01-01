# alerts/telegram.py
from __future__ import annotations

import time
import random
from typing import Any, Dict, Optional

import httpx


def _f(x: Any, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "0.000"


def _signed(x: Any, nd: int = 3) -> str:
    try:
        v = float(x)
        return f"{v:+.{nd}f}"
    except Exception:
        return f"{0.0:+.{nd}f}"


def _px(x: Any, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "0.0000"


def _lev(x: Any) -> str:
    try:
        return f"{float(x):.2f}x"
    except Exception:
        return "0.00x"


class TelegramAlerter:
    def __init__(
        self,
        token: str,
        chat_id: str,
        debug: bool = False,
        instance_name: str = "UNKNOWN",
        force_ipv4: bool = False,
        max_retries: int = 4,
        backoff_base_sec: float = 0.8,
        backoff_cap_sec: float = 6.0,
    ):
        self.token = token.strip()
        self.chat_id = str(chat_id).strip()
        self.debug = bool(debug)
        self.instance_name = str(instance_name or "UNKNOWN")

        self.force_ipv4 = bool(force_ipv4)
        self.max_retries = int(max_retries)
        self.backoff_base_sec = float(backoff_base_sec)
        self.backoff_cap_sec = float(backoff_cap_sec)

        self.base = f"https://api.telegram.org/bot{self.token}"

        timeout = httpx.Timeout(8.0, connect=6.0)

        local_addr = "0.0.0.0" if self.force_ipv4 else None
        if local_addr is not None:
            try:
                self.client = httpx.Client(
                    timeout=timeout,
                    follow_redirects=True,
                    trust_env=False,
                    local_address=local_addr,
                )
            except TypeError:
                if self.debug:
                    print("[tg-debug] httpx has no local_address support; continuing without IPv4 bind.")
                self.client = httpx.Client(timeout=timeout, follow_redirects=True, trust_env=False)
        else:
            self.client = httpx.Client(timeout=timeout, follow_redirects=True, trust_env=False)

    def _sleep_backoff(self, attempt: int) -> None:
        base = min(self.backoff_cap_sec, self.backoff_base_sec * (2 ** max(attempt - 1, 0)))
        jitter = random.uniform(0.0, 0.25 * base)
        time.sleep(base + jitter)

    def _request_with_retry(self, method: str, url: str, *, data: Optional[dict] = None) -> httpx.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.client.request(method, url, data=data)
                r.raise_for_status()
                return r
            except Exception as e:
                last_exc = e
                if self.debug:
                    print(f"[tg-debug] attempt={attempt}/{self.max_retries} err={e}")
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt)
                    continue
                break
        raise RuntimeError(f"telegram request failed after retries: {last_exc}")

    def ping(self) -> bool:
        r = self._request_with_retry("GET", f"{self.base}/getMe")
        data = r.json()
        return bool(data.get("ok"))

    def _fmt_exact(self, o: Dict[str, Any]) -> str:
        inst = self.instance_name
        sym = o.get("symbol") or o.get("sym") or "UNKNOWN"
        tier = o.get("tier", "NA")
        side = str(o.get("side", "UNKNOWN")).upper()

        ofi = o.get("ofi", 0.0)
        vpin = o.get("vpin", 0.0)
        sigma_next = o.get("sigma_next", 0.0)
        lev_cap = o.get("lev_cap", 0.0)
        spr = o.get("spread_bps", 0.0)
        micro = o.get("microprice_ticks", 0.0)

        # ✅ FIXED: unmatched parenthesis removed
        entry = o.get("entry", o.get("primary_entry", o.get("PRIMARY_ENTRY", o.get("price", 0.0))))
        stop = o.get("stop", 0.0)

        tp = o.get("tp")
        if tp is None:
            tp = o.get("tp1")
        if tp is None:
            tp = o.get("tp2", 0.0)

        vf_side = o.get("vf_side", None)
        vf_h = o.get("vf_horizon_sec", None)

        kl = o.get("kyle_lambda", None)

        lines = [
            f"🛰️ {inst}  {sym} ({tier})  {side}",
            f"   ofi={_signed(ofi, 3)}  vpin={_f(vpin, 3)}",
            f"   sigma_next={_f(sigma_next, 6)}  lev_cap~{_lev(lev_cap)}",
            f"  spread={_f(spr, 2)} bps  microprice_ticks={_signed(micro, 2)}",
            f"  PRIMARY ENTRY: {_px(entry, 4)}",
            f"   STOP: {_px(stop, 4)}",
            f"   TP: {_px(tp, 4)}",
        ]

        if vf_side is not None and vf_h is not None:
            lines.append(f"   🔔 VF: {str(vf_side).upper()} volume likely within ~{int(vf_h)}s")

        if kl is not None:
            try:
                lines.append(f"  KL: {float(kl):.1f}")
            except Exception:
                lines.append("  KL: 0.0")

        return "\n".join(lines)

    def send_entry(self, out: Dict[str, Any]) -> None:
        text = self._fmt_exact(out)
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }

        r = self._request_with_retry("POST", f"{self.base}/sendMessage", data=payload)

        if self.debug:
            try:
                print(f"[tg-debug] status={r.status_code} body={r.text[:300]}")
            except Exception:
                pass

        data = r.json()
        if not data.get("ok", False):
            raise RuntimeError(f"telegram send failed: {data}")
