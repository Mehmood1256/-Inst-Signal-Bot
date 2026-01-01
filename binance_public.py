# exchange/binance_public.py
from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

import httpx

BASE = "https://fapi.binance.com"


class BinanceFuturesPublicClient:
    """
    Public Binance Futures client with:
      - exchangeInfo caching
      - header-aware throttling via x-mbx-used-weight-1m
      - safe retries/backoff for 429 Too Many Requests

    IMPORTANT:
      This client is used in a tight loop. If throttling is too aggressive,
      it will slow the entire bot loop (api_age/loop grows) and the bot looks "quiet".
    """

    def __init__(
        self,
        timeout: float = 10.0,  # kept for backwards-compat, but we use granular timeouts below
        max_retries: int = 3,
        base_backoff_sec: float = 0.35,
        max_backoff_sec: float = 6.0,
    ):
        # Fast-fail timeouts (prevents long stalls)
        # connect: new TCP/TLS handshake
        # read: server response wait
        # pool: waiting for an available connection
        t = httpx.Timeout(connect=3.0, read=6.0, write=6.0, pool=3.0)

        # Bigger pool helps when many symbols are polled
        limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)

        self._client = httpx.Client(timeout=t, limits=limits, http2=False)

        # exchangeInfo cache
        self._exinfo_cache: Dict | None = None
        self._exinfo_cache_ts: float = 0.0
        self._exinfo_ttl_sec: float = 300.0  # 5 min (reduces weight + calls)

        # rate-limit tracking
        self._used_weight_1m: Optional[int] = None
        self._used_weight: Optional[int] = None
        self._last_throttle_ts: float = 0.0

        # backoff/retry controls
        self._max_retries = int(max_retries)
        self._base_backoff_sec = float(base_backoff_sec)
        self._max_backoff_sec = float(max_backoff_sec)

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _update_rate_headers(self, r: httpx.Response) -> None:
        """Capture Binance weight headers when present."""
        try:
            w1m = r.headers.get("x-mbx-used-weight-1m")
            if w1m is not None:
                self._used_weight_1m = int(w1m)
        except Exception:
            pass

        try:
            w = r.headers.get("x-mbx-used-weight")
            if w is not None:
                self._used_weight = int(w)
        except Exception:
            pass

    def _maybe_throttle(self) -> None:
        """
        If weight is close to Binance's 1m cap, add a small sleep to avoid 429 bursts.

        KEY CHANGE:
          Old thresholds started sleeping at w>=900 which can slow the bot loop massively
          (dozens of requests => seconds/loop).
          We only start throttling when we are actually close to the cap.
        """
        if self._used_weight_1m is None:
            return

        w = self._used_weight_1m

        # Binance Futures 1m weight limit is high; start throttling near the top only.
        sleep_s = 0.0
        if w >= 2350:
            sleep_s = 0.80
        elif w >= 2250:
            sleep_s = 0.50
        elif w >= 2150:
            sleep_s = 0.25

        now = time.time()
        # Don't sleep too often — spacing prevents turning the whole loop into molasses.
        if sleep_s > 0 and (now - self._last_throttle_ts) > 0.50:
            self._last_throttle_ts = now
            time.sleep(sleep_s + random.uniform(0.0, 0.10))

    def _request(self, method: str, path: str, *, params: dict | None = None) -> httpx.Response:
        """
        Central request wrapper:
          - header-aware throttle
          - retry/backoff for 429
          - retry/backoff for transient network errors
        """
        url = f"{BASE}{path}"
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            self._maybe_throttle()

            try:
                r = self._client.request(method, url, params=params)
                self._update_rate_headers(r)

                # 429: back off and retry
                if r.status_code == 429:
                    retry_after = 0.0
                    ra = r.headers.get("retry-after")
                    if ra:
                        try:
                            retry_after = float(ra)
                        except Exception:
                            retry_after = 0.0

                    backoff = min(self._base_backoff_sec * (2**attempt), self._max_backoff_sec)
                    backoff = max(backoff, retry_after)
                    backoff += random.uniform(0.0, 0.25)

                    if attempt >= self._max_retries:
                        r.raise_for_status()

                    time.sleep(backoff)
                    continue

                r.raise_for_status()
                return r

            except httpx.HTTPStatusError as e:
                # non-429 HTTP errors: raise immediately
                last_exc = e
                raise

            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError) as e:
                # transient transport errors: retry a few times
                last_exc = e
                if attempt >= self._max_retries:
                    raise
                backoff = min(self._base_backoff_sec * (2**attempt), self._max_backoff_sec)
                backoff += random.uniform(0.0, 0.25)
                time.sleep(backoff)

            except Exception as e:
                # other network-ish errors: retry a bit
                last_exc = e
                if attempt >= self._max_retries:
                    raise
                backoff = min(self._base_backoff_sec * (2**attempt), self._max_backoff_sec)
                backoff += random.uniform(0.0, 0.25)
                time.sleep(backoff)

        if last_exc:
            raise last_exc
        raise RuntimeError("request failed unexpectedly")

    # -----------------------------
    # Public API (same as before)
    # -----------------------------
    def exchange_info(self) -> Dict:
        now = time.time()
        if self._exinfo_cache is not None and (now - self._exinfo_cache_ts) < self._exinfo_ttl_sec:
            return self._exinfo_cache

        r = self._request("GET", "/fapi/v1/exchangeInfo")
        self._exinfo_cache = r.json()
        self._exinfo_cache_ts = now
        return self._exinfo_cache

    def list_usdt_perp_symbols(self) -> List[str]:
        info = self.exchange_info()
        out = []
        for s in info.get("symbols", []):
            if (
                s.get("contractType") == "PERPETUAL"
                and s.get("quoteAsset") == "USDT"
                and s.get("status") == "TRADING"
            ):
                out.append(s["symbol"])
        return out

    def tickers_24h(self) -> List[Dict]:
        r = self._request("GET", "/fapi/v1/ticker/24hr")
        return r.json()

    def kline(self, symbol: str, interval: str = "1m", limit: int = 300) -> List:
        r = self._request(
            "GET",
            "/fapi/v1/klines",
            params={"symbol": symbol.replace("_", ""), "interval": interval, "limit": limit},
        )
        return r.json()

    def depth(self, symbol: str, limit: int = 20) -> Dict:
        r = self._request(
            "GET",
            "/fapi/v1/depth",
            params={"symbol": symbol.replace("_", ""), "limit": limit},
        )
        data = r.json()
        return {"bids": data["bids"], "asks": data["asks"]}

    def deals(self, symbol: str, limit: int = 50) -> List:
        r = self._request(
            "GET",
            "/fapi/v1/trades",
            params={"symbol": symbol.replace("_", ""), "limit": limit},
        )
        return r.json()

    def contract_detail(self, symbol: str) -> Dict:
        info = self.exchange_info()
        sym = symbol.replace("_", "")
        for s in info.get("symbols", []):
            if s.get("symbol") == sym:
                price_filter = next(f for f in s.get("filters", []) if f.get("filterType") == "PRICE_FILTER")
                return {"priceUnit": float(price_filter["tickSize"]), "symbol": sym}
        raise ValueError(f"Symbol {symbol} not found in exchangeInfo")

    def funding_rate(self, symbol: str) -> float:
        r = self._request(
            "GET",
            "/fapi/v1/premiumIndex",
            params={"symbol": symbol.replace("_", "")},
        )
        return float(r.json()["lastFundingRate"])

    def last_used_weight_1m(self) -> Optional[int]:
        return self._used_weight_1m
