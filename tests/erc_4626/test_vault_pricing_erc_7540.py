"""Test ERC-7540 (Lagoon) vault price reading and estimation.

ERC-7540 vaults disable ``previewDeposit()``/``previewRedeem()`` on-chain,
so :py:class:`VaultPricing` must estimate prices through the
:py:class:`~eth_defi.vault.base.VaultDepositManager` abstraction
(``convertToShares()``/``convertToAssets()``) instead of the synchronous
ERC-4626 preview functions.
"""
import datetime
import os
from decimal import Decimal
from typing import cast

import pytest
from web3 import Web3

from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.classification import create_vault_instance_autodetect
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.provider.anvil import AnvilLaunch, fork_network_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_typing import HexAddress

from tradeexecutor.ethereum.vault.vault_live_pricing import VaultPricing
from tradeexecutor.ethereum.vault.vault_valuation import VaultValuator
from tradeexecutor.ethereum.vault.vault_utils import translate_vault_to_trading_pair
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.state.valuation import ValuationUpdate


JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")

pytestmark = pytest.mark.skipif(not JSON_RPC_BASE, reason="No JSON_RPC_BASE environment variable")


@pytest.fixture()
def anvil_base_fork() -> AnvilLaunch:
    """Fork Base at a pinned block so price asserts are deterministic."""
    launch = fork_network_anvil(
        JSON_RPC_BASE,
        fork_block_number=35_094_246,
    )
    try:
        yield launch
    finally:
        launch.close()


@pytest.fixture()
def web3(anvil_base_fork: AnvilLaunch) -> Web3:
    web3 = create_multi_provider_web3(
        anvil_base_fork.json_rpc_url,
        retries=1,
        default_http_timeout=(3, 250.0),
    )
    assert web3.eth.chain_id == 8453
    return web3


@pytest.fixture()
def lagoon_vault(web3: Web3) -> LagoonVault:
    """722Capital-USDC ERC-7540 vault on Base.

    https://app.lagoon.finance/vault/8453/0xb09f761cb13baca8ec087ac476647361b6314f98
    """
    vault = create_vault_instance_autodetect(
        web3,
        vault_address="0xb09f761cb13baca8ec087ac476647361b6314f98",
    )
    return cast(LagoonVault, vault)


@pytest.fixture()
def lagoon_usdc(lagoon_vault: LagoonVault) -> TradingPairIdentifier:
    pair = translate_vault_to_trading_pair(lagoon_vault)
    assert pair.is_vault()
    return pair


@pytest.fixture()
def owner() -> HexAddress:
    """Any address works for owner-scoped checks: ERC-7540 estimates ignore the owner."""
    return "0x3B95C7cD4075B72ecbC4559AF99211C2B6591b2E"


@pytest.fixture()
def vault_pricing(web3: Web3, owner: HexAddress) -> VaultPricing:
    return VaultPricing(
        web3,
        owner_address_resolver=lambda pair: owner,
    )


def test_erc_7540_vault_pricing(
    vault_pricing: VaultPricing,
    lagoon_vault: LagoonVault,
    lagoon_usdc: TradingPairIdentifier,
):
    """Price an ERC-7540 vault that has previewDeposit()/previewRedeem() disabled.

    1. Check the vault is detected as ERC-7540 (Lagoon)
    2. Estimate buy price (deposit) via the deposit manager (convertToShares)
    3. Estimate sell price (redeem) via the deposit manager (convertToAssets)
    4. Check the mid price matches buy/sell as there are no fees in conversions
    5. Check max deposit/redemption is unlimited (async request-based flow)
       and the pair reports as tradeable
    6. Check TVL reading works
    """

    # 1. Check the vault is detected as ERC-7540 (Lagoon)
    assert lagoon_vault.erc_7540
    assert not lagoon_vault.get_deposit_manager().has_synchronous_deposit()

    # 2. Estimate buy price (deposit) via the deposit manager (convertToShares)
    buy_estimate = vault_pricing.get_buy_price(
        ts=None,
        pair=lagoon_usdc,
        reserve=Decimal("100.00"),
    )
    assert buy_estimate.block_number > 0
    assert buy_estimate.mid_price == pytest.approx(1.0409669064246259)  # Forked by block mainnet

    # 3. Estimate sell price (redeem) via the deposit manager (convertToAssets)
    sell_estimate = vault_pricing.get_sell_price(
        ts=None,
        pair=lagoon_usdc,
        quantity=Decimal("100.00"),
    )
    assert sell_estimate.mid_price == pytest.approx(buy_estimate.mid_price, rel=1e-4)

    # 4. Check the mid price matches buy/sell as there are no fees in conversions
    mid_price = vault_pricing.get_mid_price(ts=None, pair=lagoon_usdc)
    assert mid_price == pytest.approx(buy_estimate.mid_price, rel=1e-4)

    # 5. Check max deposit/redemption is unlimited (async request-based flow)
    assert vault_pricing.get_max_deposit(None, lagoon_usdc) is None
    assert vault_pricing.get_max_redemption(None, lagoon_usdc) is None
    assert vault_pricing.can_deposit(None, lagoon_usdc) is True
    assert vault_pricing.can_redeem(None, lagoon_usdc) is True
    assert vault_pricing.is_tradeable(None, lagoon_usdc) is True

    # 6. Check TVL reading works
    tvl = vault_pricing.get_usd_tvl(None, lagoon_usdc)
    assert tvl > 0


def test_erc_7540_vault_valuation(
    vault_pricing: VaultPricing,
    lagoon_usdc: TradingPairIdentifier,
):
    """Value an open ERC-7540 vault position with the live share price.

    1. Create a position holding 100 vault shares
    2. Revalue the position using VaultValuator
    3. Check the new value reflects the live ERC-7540 share price
    """

    # 1. Create a position holding 100 vault shares
    position = TradingPosition(
        position_id=1,
        pair=lagoon_usdc,
        opened_at=native_datetime_utc_now(),
        last_pricing_at=native_datetime_utc_now(),
        last_token_price=1.0,
        last_reserve_price=1.0,
        last_trade_at=1.0,
        reserve_currency=lagoon_usdc.quote,
        trades={
            1: TradeExecution(
                trade_id=1,
                position_id=1,
                trade_type=TradeType.rebalance,
                opened_at=native_datetime_utc_now(),
                pair=lagoon_usdc,
                executed_at=native_datetime_utc_now(),
                executed_quantity=Decimal(100),
                planned_quantity=Decimal(100),
                planned_reserve=Decimal(100),
                planned_price=1.0,
                reserve_currency=lagoon_usdc.quote,
            )
        }
    )

    # 2. Revalue the position using VaultValuator
    valuation_model = VaultValuator(vault_pricing)
    timestamp = datetime.datetime(2029, 1, 1)
    valuation = valuation_model(ts=timestamp, position=position)

    # 3. Check the new value reflects the live ERC-7540 share price
    assert isinstance(valuation, ValuationUpdate)
    assert valuation.new_price == pytest.approx(1.0409669064246259)  # Forked by block mainnet
    assert valuation.new_value == pytest.approx(104.09669064246259)  # 100 shares * price
    assert position.get_last_valued_at() == timestamp
