"""D2 capabilities and explicitly assumed early HYPE++ history.

Actual availability always takes precedence. The bounded schedule preserves the
research assumption only before collection began; it is never a live policy.
"""

import datetime

from tradeexecutor.backtest.vault_windows import VaultWindowSchedule
from tradeexecutor.state.identifier import TradingPairIdentifier

#: Keep requests pending for the existing fourteen-day backtest assumption when
#: metadata supplies no estimate, preventing immediate reuse of unsettled cash.
SETTLEMENT_DELAY_SECONDS = 14 * 86400

#: Restrict the reconstructed history to the inspected HYPE++ deployment;
#: other D2 vaults may have different epochs and must not inherit its schedule.
HYPE_VAULT = (42161, "0x75288264fdfea8ce68e6d852696ab1ce2f3e5004")
#: Start the assumed schedule at the first available price sample so it cannot
#: imply deposit availability before the vault can be valued.
HISTORY_START = datetime.datetime(2024, 11, 20, 9, 20, 49)
#: Stop the assumption at the daily boundary after the first availability sample.
#: Using that intraday sample at the start of its own day would introduce look-ahead.
HISTORY_END = datetime.datetime(2026, 2, 12)
#: Preserve the source research's two-day deposit window every fourteen days
#: only within the observation gap bounded above; this is assumed history, not
#: evidence that the vault was open on those dates.
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
    """Prevent the early-history assumption from becoming an indefinite deposit permission.

    Once collection has begun, missing HYPE++ observations must block new deposits.
    Other vaults retain the caller's unknown-state policy because this history
    reconstruction provides no evidence about their opening schedules.
    """
    return (timestamp is not None and timestamp >= HISTORY_END
            and (pair.chain_id, pair.pool_address.lower()) == HYPE_VAULT)
