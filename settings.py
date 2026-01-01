from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class Settings:
    # ============================================================
    # INSTANCE / ORIGIN TAG (Telegram + console tag)
    # ============================================================
    BOT_INSTANCE_NAME: str = "LOCAL"  # on droplet set to "DL"

    # ============================================================
    # SYMBOL UNIVERSE (Tier-1)
    # ============================================================
    SYMBOLS: List[str] = (
        "BTC_USDT",
        "ETH_USDT",
    )

    # ============================================================
    # POLLING / LOOP
    # ============================================================
    LOOP_SLEEP_SEC: float = 1.0
    HEARTBEAT_SEC: int = 180  # 3 minutes

    # ============================================================
    # MARKET DATA
    # ============================================================
    DEPTH_LEVELS: int = 10
    KLINE_INTERVAL: str = "1m"
    KLINES_LIMIT: int = 300
    DEALS_LIMIT: int = 80

    # ============================================================
    # GARCH (Engle 1,1)
    # ============================================================
    GARCH_ALPHA: float = 0.08
    GARCH_BETA: float = 0.90

    # Vol filters: block chaos AND dead chop
    VOL_KILL: float = 0.020
    VOL_FLOOR: float = 0.00014
    VOL_FLOOR_OFI_MULT: float = 1.35
    VOL_FLOOR_MICRO_MULT: float = 1.35

    # ============================================================
    # RISK & LEVERAGE
    # ============================================================
    TARGET_RISK_FRAC: float = 0.0025
    L_MAX: float = 15.0
    K_SL: float = 3.0

    TP1_R: float = 0.60
    TP2_R: float = 1.20

    # ============================================================
    # MICROSTRUCTURE SIGNALS (CORE STRATEGY)
    # ============================================================
    BOOK_LEVELS_K: int = 10

    # base thresholds (CAND uses these; T1/T2 apply multipliers below)
    OFI_THR: float = 0.24
    MICRO_TICKS_MIN: float = 0.12

    ENTER_PERSIST_N: int = 2

    # ============================================================
    # VPIN (volume buckets)
    # ============================================================
    VPIN_BUCKET_NOTIONAL_USD: float = 75_000.0
    VPIN_LOOKBACK_BUCKETS: int = 50

    VPIN_MIN: float = 0.20
    VPIN_MIN_BUCKETS_READY: int = 2

    # warmup gate stays on
    VPIN_HARD_WARMUP_BLOCK: bool = True

    # ============================================================
    # VPIN BUCKETS (tiered + volume-adaptive)
    # ============================================================
    VPIN_BUCKET_T1_USD: float = 120_000.0
    VPIN_BUCKET_T2_USD: float = 80_000.0
    VPIN_BUCKET_T3_USD: float = 50_000.0

    VPIN_BUCKET_VOL_REF_USD: float = 25_000_000.0
    VPIN_BUCKET_SCALE_MIN: float = 0.50
    VPIN_BUCKET_SCALE_MAX: float = 3.00

    # ============================================================
    # LIQUIDITY GATE
    # ============================================================
    REL_SPREAD_MAX_BPS: float = 8.0

    SPREAD_EWMA_ENABLED: bool = True
    SPREAD_EWMA_ALPHA: float = 0.05
    SPREAD_EWMA_MIN_SAMPLES: int = 60
    SPREAD_EWMA_MULT: float = 1.70

    # ============================================================
    # REGIME LOCK / ANTI-FLIP
    # ============================================================
    REGIME_LOCK_SEC: int = 300
    FLIP_PERSIST_N: int = 3

    SIGNAL_COOLDOWN_SEC: int = 240
    SIGNAL_FLIP_COOLDOWN_SEC: int = 900
    SIGNAL_MIN_MOVE_BPS_TO_RESIGNAL: float = 20.0

    # ============================================================
    # PULLBACK SNIPER FILTER  (DISABLED — FINAL FIX REQUESTED)
    # ============================================================
    PULLBACK_ENABLED: bool = False
    PULLBACK_WINDOW_SEC: int = 60
    PULLBACK_MIN_BPS: float = 8.0

    # ============================================================
    # ENTRY CONFIRMATION
    # ============================================================
    ENTRY_CONFIRM_ENABLED: bool = True
    ENTRY_CONFIRM_WINDOW_SEC: int = 45

    ENTRY_MAX_CHASE_BPS_MIN: float = 6.0
    ENTRY_MAX_CHASE_BPS_MAX: float = 12.0
    ENTRY_MAX_CHASE_BPS_SPR_MULT: float = 3.0

    ENTRY_CONFIRM_MICRO_FRAC: float = 0.70
    ENTRY_CONFIRM_OFI_FRAC: float = 0.70
    ENTRY_CONFIRM_MIN_BOOKS: int = 3
    SPREAD_CONFIRM_MULT: float = 1.15
    DEBUG_ENTRY_CONFIRM: bool = True

    # ============================================================
    # MERTON-LITE JUMP RISK FILTER (ENTRY ONLY)
    # ============================================================
    JUMP_RISK_ENABLED: bool = True
    JUMP_Z_WINDOW: int = 300
    JUMP_SCORE_THR: float = 2.20
    JUMP_CANCEL_SETUP_IF_HIGH: bool = False

    JUMP_W_VPIN: float = 0.45
    JUMP_W_SPREAD: float = 0.30
    JUMP_W_ABS_OFI: float = 0.15
    JUMP_W_ABS_MICRO: float = 0.10

    # ============================================================
    # ADAPTIVE PER-COIN SCALING
    # ============================================================
    ADAPTIVE_ENABLED: bool = True
    ADAPT_WINDOW_SAMPLES: int = 300
    ADAPT_MIN_SAMPLES: int = 120
    ADAPT_REFRESH_SEC: int = 300

    OFI_Q: float = 0.70
    MICRO_Q: float = 0.70
    VPIN_Q: float = 0.70

    OFI_THR_MIN: float = 0.15
    OFI_THR_MAX: float = 0.45

    MICRO_TICKS_MIN_MIN: float = 0.06
    MICRO_TICKS_MIN_MAX: float = 0.25

    VPIN_MIN_MIN: float = 0.20
    VPIN_MIN_MAX: float = 0.80

    # ============================================================
    # TIER LOOSENING (T1/T2 only; CAND uses base cfg)
    # ============================================================
    # Loosen a bit so BTC/ETH actually fire.
    T1_OFI_MULT: float = 0.75
    T1_MICRO_MULT: float = 0.75
    T1_VPIN_MIN: float = 0.15
    T1_PULLBACK_MIN_BPS: float = 0.0  # irrelevant because pullback disabled

    T2_OFI_MULT: float = 0.85
    T2_MICRO_MULT: float = 0.85
    T2_VPIN_MIN: float = 0.18
    T2_PULLBACK_MIN_BPS: float = 0.0  # irrelevant because pullback disabled

    # ============================================================
    # TIERED UNIVERSE (slightly more coins, not a burden)
    # ============================================================
    WATCHLIST_TOP_N: int = 150          # was 120
    T2_MIN_QUOTE_VOL: float = 25_000_000.0
    T2_SPREAD_MAX_BPS: float = 8.0
    T2_MAX_CANDIDATES: int = 45         # was 35
    T2_MAX_ACTIVE: int = 35             # was 30

    T2_CONFIRM_VPIN_MIN: float = 0.45
    T2_CONFIRM_SIGMA_MAX: float = 0.0025

    TIER_REFRESH_SEC: int = 300
    T2_CONFIRM_SEC: int = 60

    T2_FLIP_WINDOW_SEC: int = 30 * 60
    T2_MAX_FLIPS_IN_WINDOW: int = 2

    # ============================================================
    # VOLUME FORECAST (gate)
    # ============================================================
    VOLUME_FORECAST_ENABLED: bool = True

    VF_HAWKES_TAU_SEC: float = 30.0
    VF_HAWKES_Z_WINDOW: int = 240
    VF_BURST_Z_THR: float = 1.5
    VF_P_DIR_THR: float = 0.65

    VF_L2_BAND_BPS: float = 10.0
    VF_L2_EMA_ALPHA: float = 0.03

    VF_W_HAWKES: float = 0.55
    VF_W_L2: float = 0.45
    VF_COMBINE_THR: float = 0.63
    VF_COOLDOWN_SEC: int = 60

    VF_ATTACH_MAX_AGE_SEC: int = 240
    VF_PRINT_STANDALONE: bool = False

    VF_GATE_ENABLED: bool = True
    VF_GATE_ALLOW_NO_HINT: bool = True
    VF_GATE_REQUIRE_MATCH: bool = True

    # ============================================================
    # KYLE'S LAMBDA (SNIPER OVERLAY)
    # ============================================================
    KYLE_LAMBDA_ENABLED: bool = True
    KL_WINDOW_SEC: int = 60
    KL_EMA_ALPHA: float = 0.20
    KL_MIN_NOTIONAL_USD: float = 20_000.0
    KL_LAMBDA_MIN_BPS_PER_MUSD: float = 65.0
    KL_FLOW_SIGN_MIN_FRAC: float = 0.05
    KL_MIN_MID_MOVE_BPS: float = 6.0

    # ============================================================
    # TELEGRAM ALERTS
    # ============================================================
    TELEGRAM_ENABLED: bool = True
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = "5092611265"
    TELEGRAM_DEBUG: bool = True
