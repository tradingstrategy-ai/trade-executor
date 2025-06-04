"""Backfill share price records on statistics when it was missing."""
from eth_defi.erc_4626.vault import ERC4626Vault
from tradeexecutor.state.state import State


def retrofit_share_price(
    state: State,
    vault: ERC4626Vault,
):

    statistics = state.stats