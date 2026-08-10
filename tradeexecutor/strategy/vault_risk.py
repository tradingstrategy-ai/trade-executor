"""Vault risk guards used during trading-universe construction."""

from typing import TYPE_CHECKING

from tradeexecutor.state.identifier import IGNORE_REASON_BLACKLISTED_VAULT, TradingPairIdentifier

if TYPE_CHECKING:
    from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


#: Pair ``other_data`` flag for a vault the metadata producer classifies as blacklisted.
BLACKLISTED_VAULT_DIAGNOSTIC_FLAG = "blacklisted_vault"


def is_blacklisted_vault(pair: TradingPairIdentifier) -> bool:
    """Check whether producer metadata classifies a vault as blacklisted."""
    if not pair.is_vault():
        return False

    risk_level = pair.get_vault_risk_level()
    return isinstance(risk_level, str) and risk_level.casefold() == "blacklisted"


def mark_blacklisted_vaults_ignored(
    strategy_universe: "TradingStrategyUniverse",
) -> list[TradingPairIdentifier]:
    """Retain blacklisted vaults for diagnostics but exclude them from allocation.

    This guard runs automatically when :py:meth:`TradingStrategyUniverse.create_from_dataset`
    constructs a universe. The diagnostic flag is retained even if another
    constraint has already set an ignore reason.
    """
    flagged = []
    for pair in strategy_universe.iterate_pairs():
        if not is_blacklisted_vault(pair):
            continue

        pair.other_data[BLACKLISTED_VAULT_DIAGNOSTIC_FLAG] = True
        if pair.get_ignore_reason() is None:
            pair.set_ignore_reason(IGNORE_REASON_BLACKLISTED_VAULT)
        flagged.append(pair)

    return flagged
