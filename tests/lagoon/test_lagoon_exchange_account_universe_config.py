"""Exchange-account guard whitelist coverage."""

from types import SimpleNamespace

from eth_defi.token import USDC_NATIVE_TOKEN

from tradeexecutor.ethereum.lagoon.universe_config import _collect_chain_token_addresses
from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pair
from tradeexecutor.state.identifier import AssetIdentifier


def test_exchange_account_guard_whitelists_only_quote_asset():
    """Do not whitelist a synthetic Lighter account identity as an ERC-20.

    1. Create the public Lighter exchange-account pair used by a strategy.
    2. Collect Lagoon guard token addresses with strict asset whitelisting.
    3. Verify only native Ethereum USDC is eligible for the guard token list.
    """
    # 1. Create the public Lighter exchange-account pair used by a strategy.
    usdc = AssetIdentifier(
        chain_id=1,
        address=USDC_NATIVE_TOKEN[1],
        token_symbol="USDC",
        decimals=6,
    )
    pair = create_lighter_exchange_account_pair(usdc, account_index=123)

    # 2. Collect Lagoon guard token addresses with strict asset whitelisting.
    addresses = _collect_chain_token_addresses(
        SimpleNamespace(iterate_pairs=lambda: [pair]),
        all_chain_ids={1},
        any_asset=False,
    )

    # 3. Verify only native Ethereum USDC is eligible for the guard token list.
    assert addresses == {1: {USDC_NATIVE_TOKEN[1]}}
