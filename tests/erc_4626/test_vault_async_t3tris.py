"""Exercise T3tris through the existing live checks and guarded settlement flow.

First USDC and the fixed Arbitrum block match eth-defi's T3tris integration
tests. Only the strategy Safe is funded; the curator must settle redemptions
using its existing liquidity. Oracle/curator impersonation is confined to the
adapter's Anvil settlement driver.
"""

import logging
import os
from collections.abc import Iterator
from decimal import Decimal

import pytest
from web3 import Web3

from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.erc_4626.vault import ERC4626Vault
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import LagoonConfig, LagoonDeploymentParameters, deploy_automated_lagoon_vault
from eth_defi.erc_4626.vault_protocol.t3tris.vault import T3trisVault
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fork_network_anvil, fund_erc20_on_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.testing.fork_blocks import ARBITRUM_MIDNIGHT_BLOCK
from eth_defi.vault.base import VaultSpec

from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.lagoon.execution import LagoonExecution
from tradeexecutor.ethereum.lagoon.tx import LagoonTransactionBuilder
from tradeexecutor.ethereum.vault.vault_live_pricing import VaultPricing
from tradeexecutor.ethereum.vault.vault_utils import translate_vault_to_trading_pair
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.reverse_universe import create_universe_from_trading_pair_identifiers
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType, ExchangeUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
FIRST_USDC = "0x98e43a491a464f0886bc5e57207c340bbed0d01f"
GAMI_USDC = "0x9984ad74c5fb6bec3888e14b4e453707d3be7f8f"
pytestmark = pytest.mark.skipif(not JSON_RPC_ARBITRUM, reason="Set JSON_RPC_ARBITRUM to run T3tris fork tests")


@pytest.fixture()
def web3() -> Iterator[Web3]:
    """Use a fresh process for each deployment-heavy guarded lifecycle test."""
    launch = fork_network_anvil(JSON_RPC_ARBITRUM, fork_block_number=ARBITRUM_MIDNIGHT_BLOCK)
    try:
        yield create_multi_provider_web3(launch.json_rpc_url, retries=1)
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.mark.timeout(300)
def test_t3tris_guarded_async_round_trip(web3: Web3) -> None:
    """Shared routing must persist and claim T3tris tickets through Safe custody.

    1. Deploy ordinary Lagoon custody with an address/feature vault descriptor.
    2. Request a deposit through the executor and hold its queue.
    3. Reload pending State, settle the epoch and claim the shares.
    4. Request redemption and confirm escrow does not release reserve cash.
    5. Reload, settle and claim once; repeated resolution leaves balances alone.
    """
    # 1. Deploy ordinary Lagoon custody with an address/feature vault descriptor.
    target = create_vault_instance(web3, FIRST_USDC, features={ERC4626Feature.t3tris_like})
    assert isinstance(target, T3trisVault)
    manager = target.get_deposit_manager()
    pair = translate_vault_to_trading_pair(target)
    exchange = Exchange(chain_id=ChainId.arbitrum, chain_slug="arbitrum", exchange_id=1,
        exchange_slug="t3tris", address=pair.exchange_address,
        exchange_type=ExchangeType.erc_4626_vault, pair_count=1)
    exchanges = ExchangeUniverse({1: exchange})
    pairs = create_universe_from_trading_pair_identifiers([pair], exchanges)
    universe = TradingStrategyUniverse(data_universe=Universe(
        time_bucket=TimeBucket.not_applicable, chains={ChainId.arbitrum},
        pairs=pairs, exchange_universe=exchanges), reserve_assets=[pair.quote])
    pair = next(universe.iterate_pairs())
    signer = HotWallet.create_for_testing(web3, test_account_n=1, eth_amount=100)
    signer.sync_nonce(web3)
    deployment = deploy_automated_lagoon_vault(web3=web3, deployer=signer, config=LagoonConfig(
        parameters=LagoonDeploymentParameters(underlying=target.denomination_token.address,
            name="T3tris executor test", symbol="T3T"),
        safe_owners=[signer.address], safe_threshold=1, asset_manager=signer.address,
        any_asset=True, erc_4626_vaults=[ERC4626Vault(web3, VaultSpec(42161, FIRST_USDC),
            features={ERC4626Feature.t3tris_like})], use_forge=False, from_the_scratch=False))
    custody = deployment.vault
    execution = LagoonExecution(vault=custody, tx_builder=LagoonTransactionBuilder(custody, signer),
        mainnet_fork=True, confirmation_block_count=0)
    routing = execution.create_default_routing_model(universe)
    pricing = GenericPricing(EthereumPairConfigurator(web3, universe, execution_model=execution))
    route = routing.create_routing_state(universe, execution.get_routing_state_details())
    usdc = target.denomination_token
    fund_erc20_on_anvil(web3, usdc.address, custody.safe_address, usdc.convert_to_raw(Decimal(100)))
    state = State()
    state.portfolio.initialise_reserves(pair.quote, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(pair.quote, Decimal(100), "Funded test Safe")
    assert pricing.check_deposit(None, pair).can_deposit
    assert pricing.check_redemption(None, pair).can_redeem

    # 2. Request a deposit through the executor and hold its queue.
    pm = PositionManager(native_datetime_utc_now(), universe, state, pricing)
    buy = pm.open_spot(pair, value=50.0)[0]
    execution.execute_trades(native_datetime_utc_now(), state, [buy], routing, route, check_balances=True)
    assert buy.get_status() == TradeStatus.vault_settlement_pending
    assert not pricing.check_deposit(None, pair).can_deposit
    assert state.portfolio.get_cash() == pytest.approx(50)
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(50)
    assert execution.resolve_pending_vault_settlements(state=state, ts=native_datetime_utc_now()) == []

    # 3. Reload pending State, settle the epoch and claim the shares.
    state = State.from_json(state.to_json())
    buy = state.portfolio.get_trade_by_id(buy.trade_id)
    ticket = manager.reconstruct_deposit_ticket(buy.other_data)
    manager.force_settle(ticket)
    assert not pricing.check_deposit(None, pair).can_deposit
    assert execution.resolve_pending_vault_settlements(state=state, ts=native_datetime_utc_now()) == [buy]
    position = state.portfolio.get_position_by_id(buy.position_id)
    assert buy.is_success() and position.is_open()
    assert position.get_quantity() == target.share_token.fetch_balance_of(custody.safe_address)
    assert pricing.check_deposit(None, pair).can_deposit
    assert buy.blockchain_transactions[-1].contract_address.lower() == custody.trading_strategy_module_address.lower()

    # 4. Request redemption and confirm escrow does not release reserve cash.
    pm = PositionManager(native_datetime_utc_now(), universe, state, pricing)
    sell = pm.close_all()[0]
    execution.execute_trades(native_datetime_utc_now(), state, [sell], routing, route, check_balances=True)
    assert sell.get_status() == TradeStatus.vault_settlement_pending
    assert target.share_token.fetch_balance_of(custody.safe_address) == 0
    assert not pricing.check_redemption(None, pair).can_redeem
    assert state.portfolio.get_cash() == pytest.approx(50)
    assert execution.resolve_pending_vault_settlements(state=state, ts=native_datetime_utc_now()) == []

    # 5. Reload, settle and claim once; repeated resolution leaves balances alone.
    state = State.from_json(state.to_json())
    sell = state.portfolio.get_trade_by_id(sell.trade_id)
    manager.force_settle(manager.reconstruct_redemption_ticket(sell.other_data))
    assert execution.resolve_pending_vault_settlements(state=state, ts=native_datetime_utc_now()) == [sell]
    assert sell.is_success()
    assert state.portfolio.get_position_by_id(sell.position_id).is_closed()
    assert state.portfolio.get_cash() == pytest.approx(float(usdc.fetch_balance_of(custody.safe_address)))
    assert state.portfolio.get_cash() == pytest.approx(100, abs=1)
    before = usdc.fetch_balance_of(custody.safe_address)
    assert execution.resolve_pending_vault_settlements(state=state, ts=native_datetime_utc_now()) == []
    assert usdc.fetch_balance_of(custody.safe_address) == before
    assert sell.blockchain_transactions[-1].contract_address.lower() == custody.trading_strategy_module_address.lower()


def test_t3tris_live_checks_refuse_restricted_requests(web3: Web3) -> None:
    """Owner-aware checks must refuse restricted requests before broadcasting.

    1. Resolve a public vault and a deposit-disabled, withdrawal-whitelisted vault.
    2. Check permitted requests and both independent restricted operations.
    3. Check missing custody context is refused without sending transactions.
    """
    # 1. Resolve a public vault and a deposit-disabled, withdrawal-whitelisted vault.
    owner = web3.eth.accounts[8]
    public = translate_vault_to_trading_pair(create_vault_instance(web3, FIRST_USDC,
        features={ERC4626Feature.t3tris_like}))
    restricted = translate_vault_to_trading_pair(create_vault_instance(web3, GAMI_USDC,
        features={ERC4626Feature.t3tris_like}))
    pricing = VaultPricing(web3, owner_address_resolver=lambda pair: owner)
    block = web3.eth.block_number

    # 2. Check permitted requests and both independent restricted operations.
    assert pricing.check_deposit(None, public).can_deposit
    assert pricing.check_redemption(None, public).can_redeem
    assert pricing.get_max_deposit(None, public) is None
    assert pricing.get_max_redemption(None, public) is None
    assert not pricing.check_deposit(None, restricted).can_deposit
    assert not pricing.check_redemption(None, restricted).can_redeem

    # 3. Check missing custody context is refused without sending transactions.
    unknown = VaultPricing(web3)
    assert not unknown.check_deposit(None, public).can_deposit
    assert not unknown.check_redemption(None, public).can_redeem
    assert web3.eth.block_number == block
