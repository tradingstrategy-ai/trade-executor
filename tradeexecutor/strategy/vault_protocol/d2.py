"""D2 capabilities and explicitly assumed early HYPE++ history.

Actual availability always takes precedence. The bounded schedule preserves the
research assumption only before collection began; it is never a live policy.
"""

import datetime

from tradeexecutor.backtest.vault_windows import VaultWindowSchedule
from tradeexecutor.state.identifier import TradingPairIdentifier

#: Existing conservative minimum delay for D2 historical requests without an estimate.
SETTLEMENT_DELAY_SECONDS = 14 * 86400

#: Arbitrum HYPE++ identity whose early observation gap was inspected.
HYPE_VAULT = (42161, "0x75288264fdfea8ce68e6d852696ab1ce2f3e5004")
#: Beginning of actual price coverage; no tradability is fabricated before it.
HISTORY_START = datetime.datetime(2024, 11, 20, 9, 20, 49)
#: First populated availability observation becomes usable at the next daily boundary.
HISTORY_END = datetime.datetime(2026, 2, 12)
#: Original research epoch assumption: fourteen-day cadence, two open days.
EARLY_SCHEDULE = VaultWindowSchedule(
    cadence=datetime.timedelta(days=14),
    open_duration=datetime.timedelta(days=2),
    anchor=datetime.datetime(2025, 3, 30),
)


def early_window(pair: TradingPairIdentifier, timestamp: datetime.datetime | None) -> VaultWindowSchedule | None:
    """Return the early HYPE++ schedule for a naive UTC decision time, if covered.

    Call only when the historical flag, cap and restriction reason are unknown;
    this helper checks identity and dates, not the observation's contents.
    """
    if timestamp is not None and (pair.chain_id, pair.pool_address.lower()) == HYPE_VAULT and HISTORY_START <= timestamp < HISTORY_END:
        return EARLY_SCHEDULE
    return None


def unknown_deposit_is_closed(pair: TradingPairIdentifier, timestamp: datetime.datetime | None) -> bool:
    """Refuse missing later HYPE++ evidence without changing other D2 histories."""
    return (timestamp is not None and timestamp >= HISTORY_END
            and (pair.chain_id, pair.pool_address.lower()) == HYPE_VAULT)
