# exchange/binance_ws.py
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Dict, List, Optional, Set, Tuple

import websockets

FSTREAM_WS = "wss://fstream.binance.com/ws"


def _to_stream_symbol(sym: str) -> str:
    # BTC_USDT -> btcusdt
    return sym.replace("_", "").lower()


def _streams_for_symbol(sym: str, depth_levels: int) -> List[str]:
    s = _to_stream_symbol(sym)
    return [
        f"{s}@depth{depth_levels}@100ms",
        f"{s}@aggTrade",
    ]


@dataclass
class WSStats:
    depth_msgs: int = 0
    trade_msgs: int = 0
    reconnects: int = 0
    last_msg_ts: float = 0.0
    last_connect_ts: float = 0.0
    last_err: str = ""


class BinanceFuturesWSHub:
    """
    Stable Binance Futures WS hub with dynamic subscribe/unsubscribe.

    FIX: command queue is now a thread-safe Queue (not asyncio.Queue created outside loop),
    and update_symbols works even before the async loop is fully running.
    """

    def __init__(self, symbols: List[str], depth_levels: int = 10):
        self.depth_levels = int(depth_levels)

        self._lock = threading.Lock()
        self._latest_book: Dict[str, Dict[str, List[List[str]]]] = {}
        self._trade_buf: Dict[str, List[dict]] = {}

        # per-symbol freshness timestamps
        self._last_book_ts: Dict[str, float] = {}
        self._last_trade_ts: Dict[str, float] = {}

        self.stats = WSStats(last_msg_ts=time.time(), last_connect_ts=0.0, last_err="")

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._want_syms: Set[str] = set(symbols)
        self._subscribed_streams: Set[str] = set()

        # thread-safe command queue: ("SET", [streams...])
        self._cmd_q: "Queue[Tuple[str, List[str]]]" = Queue()

    # -------------------------
    # Public API
    # -------------------------
    @property
    def symbols(self) -> List[str]:
        with self._lock:
            return sorted(self._want_syms)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()

    def update_symbols(self, symbols: List[str]) -> None:
        """
        Update subscription set without restarting the socket.
        Safe to call anytime.
        """
        new_set = set(symbols)
        with self._lock:
            self._want_syms = new_set

        want_streams: Set[str] = set()
        for sym in new_set:
            want_streams.update(_streams_for_symbol(sym, self.depth_levels))

        self._cmd_q.put(("SET", sorted(want_streams)))

    def get_latest_book(self, symbol: str) -> Optional[Dict[str, List[List[str]]]]:
        with self._lock:
            b = self._latest_book.get(symbol)
            if not b:
                return None
            return {"bids": b["bids"], "asks": b["asks"]}

    def pop_trades(self, symbol: str, max_n: int = 200) -> List[dict]:
        with self._lock:
            buf = self._trade_buf.get(symbol, [])
            if not buf:
                return []
            if len(buf) <= max_n:
                out = buf[:]
                self._trade_buf[symbol] = []
                return out
            out = buf[:max_n]
            self._trade_buf[symbol] = buf[max_n:]
            return out

    def last_book_age(self, symbol: str) -> float:
        with self._lock:
            ts = self._last_book_ts.get(symbol, 0.0)
        if ts <= 0:
            return float("inf")
        return max(0.0, time.time() - ts)

    def last_trade_age(self, symbol: str) -> float:
        with self._lock:
            ts = self._last_trade_ts.get(symbol, 0.0)
        if ts <= 0:
            return float("inf")
        return max(0.0, time.time() - ts)

    # -------------------------
    # Internals
    # -------------------------
    def _run_thread(self) -> None:
        asyncio.run(self._run_loop())

    async def _send(self, ws, payload: dict) -> None:
        await ws.send(json.dumps(payload))

    async def _apply_set(self, ws, want_streams: Set[str]) -> None:
        # subscribe missing
        to_sub = sorted(want_streams - self._subscribed_streams)
        if to_sub:
            await self._send(
                ws,
                {"method": "SUBSCRIBE", "params": to_sub, "id": int(time.time() * 1000) % 1_000_000},
            )
            self._subscribed_streams.update(to_sub)

        # unsubscribe extra
        to_unsub = sorted(self._subscribed_streams - want_streams)
        if to_unsub:
            await self._send(
                ws,
                {"method": "UNSUBSCRIBE", "params": to_unsub, "id": int(time.time() * 1000) % 1_000_000},
            )
            for x in to_unsub:
                self._subscribed_streams.discard(x)

    async def _run_loop(self) -> None:
        backoff = 1.0

        # initial desired streams
        with self._lock:
            initial_syms = set(self._want_syms)
        want_streams: Set[str] = set()
        for sym in initial_syms:
            want_streams.update(_streams_for_symbol(sym, self.depth_levels))

        while not self._stop_evt.is_set():
            try:
                self.stats.last_connect_ts = time.time()
                self.stats.last_err = ""
                async with websockets.connect(
                    FSTREAM_WS,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_queue=4096,
                ) as ws:
                    self._subscribed_streams = set()
                    await self._apply_set(ws, want_streams)
                    backoff = 1.0

                    while not self._stop_evt.is_set():
                        # process all queued commands
                        while True:
                            try:
                                cmd, params = self._cmd_q.get_nowait()
                            except Empty:
                                break
                            if cmd == "SET":
                                want_streams = set(params)
                                await self._apply_set(ws, want_streams)

                        # recv with timeout (keeps loop alive)
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue

                        self.stats.last_msg_ts = time.time()
                        self._handle_message(raw)

            except Exception as e:
                self.stats.last_err = f"{type(e).__name__}: {e}"
                self.stats.reconnects += 1
                await asyncio.sleep(min(backoff, 15.0))
                backoff = min(backoff * 2.0, 15.0)

    def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return

        # subscription acks: {"result":null,"id":...}
        if "result" in msg and "id" in msg:
            return

        stream = msg.get("stream")
        data = msg.get("data")

        # If using /ws with SUBSCRIBE, messages may come without wrapper
        if stream is None and data is None:
            data = msg

        if not data:
            return

        etype = data.get("e")
        sym_raw = data.get("s")
        if not sym_raw:
            return

        symbol = sym_raw.upper().replace("USDT", "_USDT")
        now = time.time()

        if etype == "depthUpdate":
            bids = data.get("b", [])[: self.depth_levels]
            asks = data.get("a", [])[: self.depth_levels]
            if not bids or not asks:
                return
            with self._lock:
                self._latest_book[symbol] = {"bids": bids, "asks": asks}
                self._last_book_ts[symbol] = now
            self.stats.depth_msgs += 1
            return

        if etype == "aggTrade":
            try:
                tid = int(data.get("a", 0))
                price = str(data.get("p"))
                qty = str(data.get("q"))
                is_buyer_maker = bool(data.get("m", False))
                quote_qty = float(price) * float(qty)
            except Exception:
                return

            t = {
                "id": tid,
                "price": price,
                "qty": qty,
                "quoteQty": quote_qty,
                "isBuyerMaker": is_buyer_maker,
            }

            with self._lock:
                buf = self._trade_buf.get(symbol)
                if buf is None:
                    buf = []
                    self._trade_buf[symbol] = buf
                buf.append(t)
                self._last_trade_ts[symbol] = now

                if len(buf) > 2000:
                    del buf[:1000]

            self.stats.trade_msgs += 1
            return
