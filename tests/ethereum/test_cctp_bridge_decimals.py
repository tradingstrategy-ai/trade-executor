"""Regression test for CCTP bridge USDC decimal resolution.

In a vault-only trading universe there is no DEX token metadata, so the
USDC token decimals default to 18 instead of the real 6. Previously the
synthetic CCTP bridge pair inherited those wrong 18 decimals, and the
bridge routing converted the burn amount with ``10 ** 18`` instead of
``10 ** 6`` — feeding a 10**12x-too-large amount into ``depositForBurn``,
which reverts with ``ERC20: transfer amount exceeds balance``.

Native USDC is always 6 decimals on every CCTP chain, so the generated
bridge pair must carry 6-decimal USDC regardless of poisoned universe
metadata.
"""

import pytest
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType, ExchangeUniverse

from eth_defi.token import USDC_NATIVE_TOKEN

from tradeexecutor.ethereum.cctp.bridge_universe import (
    generate_primary_to_satellite_cctp_bridge_universe,
)
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.strategy.trading_strategy_universe import create_pair_universe_from_code


#: Wrong decimals a vault-only universe assigns to USDC (no DEX metadata).
POISONED_DECIMALS = 18


@pytest.fixture()
def poisoned_usdc_arbitrum() -> AssetIdentifier:
    """Arbitrum USDC with the wrong 18 decimals (vault-only universe)."""
    return AssetIdentifier(
        chain_id=ChainId.arbitrum.value,
        address=USDC_NATIVE_TOKEN[ChainId.arbitrum.value],
        token_symbol="USDC",
        decimals=POISONED_DECIMALS,
    )


@pytest.fixture()
def poisoned_usdc_base() -> AssetIdentifier:
    """Base USDC with the wrong 18 decimals (vault-only universe)."""
    return AssetIdentifier(
        chain_id=ChainId.base.value,
        address=USDC_NATIVE_TOKEN[ChainId.base.value],
        token_symbol="USDC",
        decimals=POISONED_DECIMALS,
    )


def _vault_pair(
    quote: AssetIdentifier,
    vault_address: str,
    symbol: str,
    internal_id: int,
    internal_exchange_id: int,
) -> TradingPairIdentifier:
    """Build a vault pair whose quote is the (poisoned) USDC asset."""
    share_token = AssetIdentifier(
        chain_id=quote.chain_id,
        address=vault_address,
        token_symbol=symbol,
        decimals=18,
    )
    return TradingPairIdentifier(
        base=share_token,
        quote=quote,
        pool_address=vault_address,
        exchange_address=vault_address,
        internal_id=internal_id,
        internal_exchange_id=internal_exchange_id,
        fee=0.0,
        kind=TradingPairKind.vault,
        exchange_name=symbol,
        other_data={"vault_protocol": "ipor_fusion"},
    )


def test_generated_bridge_pair_uses_native_usdc_decimals(
    poisoned_usdc_arbitrum: AssetIdentifier,
    poisoned_usdc_base: AssetIdentifier,
):
    """Generated CCTP bridge pairs must carry 6-decimal native USDC.

    Reproduces the vault-only universe scenario where USDC metadata is
    poisoned to 18 decimals and verifies the synthetic bridge pair is
    immune to it.

    1. Build a vault-only pair universe on Arbitrum (primary) and Base
       (satellite) where both USDC assets report 18 decimals.
    2. Generate the synthetic forward CCTP bridge universe.
    3. Assert one bridge pair was generated for Arbitrum -> Base.
    4. Assert both legs of the bridge pair use 6-decimal USDC, so the
       burn amount is converted correctly (no 10**12x over-burn).
    """

    # 1. Build a vault-only pair universe with poisoned 18-decimal USDC
    arb_vault = _vault_pair(
        poisoned_usdc_arbitrum, "0x0000000000000000000000000000000000000011", "vaultARB",
        internal_id=1, internal_exchange_id=101,
    )
    base_vault = _vault_pair(
        poisoned_usdc_base, "0x0000000000000000000000000000000000000022", "vaultBASE",
        internal_id=2, internal_exchange_id=102,
    )
    pair_universe = create_pair_universe_from_code(ChainId.arbitrum, [arb_vault, base_vault])

    exchange_universe = ExchangeUniverse(
        exchanges={
            ex.exchange_id: ex
            for ex in [
                Exchange(
                    chain_id=ChainId.arbitrum,
                    chain_slug="arbitrum",
                    exchange_id=101,
                    exchange_slug="vault-arb",
                    address="0x0000000000000000000000000000000000000011",
                    exchange_type=ExchangeType.erc_4626_vault,
                    pair_count=1,
                ),
                Exchange(
                    chain_id=ChainId.base,
                    chain_slug="base",
                    exchange_id=102,
                    exchange_slug="vault-base",
                    address="0x0000000000000000000000000000000000000022",
                    exchange_type=ExchangeType.erc_4626_vault,
                    pair_count=1,
                ),
            ]
        }
    )

    # 2. Generate the synthetic forward CCTP bridge universe
    result = generate_primary_to_satellite_cctp_bridge_universe(
        pairs=pair_universe,
        exchange_universe=exchange_universe,
        reserve_asset=poisoned_usdc_arbitrum,
        primary_chain=ChainId.arbitrum,
    )

    # 3. Exactly one forward bridge pair: Arbitrum -> Base
    assert len(result.generated_pairs) == 1
    bridge_pair = result.generated_pairs[0]
    assert bridge_pair.is_cctp_bridge()
    assert bridge_pair.base.chain_id == ChainId.base.value
    assert bridge_pair.quote.chain_id == ChainId.arbitrum.value

    # 4. Both legs must use 6-decimal native USDC despite the poisoned universe
    assert bridge_pair.quote.decimals == 6, "Primary (burn) USDC must be 6 decimals"
    assert bridge_pair.base.decimals == 6, "Satellite (mint) USDC must be 6 decimals"
