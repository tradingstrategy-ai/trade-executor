"""Test Enzyme accounting corrections can be applied."""
import datetime
import secrets
from decimal import Decimal

import pytest
import flaky
from eth_account import Account
from hexbytes import HexBytes

from eth_defi.enzyme.erc20 import prepare_transfer
from eth_defi.enzyme.vault import Vault
from eth_defi.hotwallet import HotWallet
from eth_defi.middleware import construct_sign_and_send_raw_middleware_anvil
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_typing import HexAddress

from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2Routing
from tradeexecutor.state.balance_update import BalanceUpdateCause, BalanceUpdatePositionType
from tradingstrategy.pair import PandasPairUniverse
from web3 import Web3
from web3.contract import Contract

from eth_defi.event_reader.reorganisation_monitor import create_reorganisation_monitor

from tradeexecutor.ethereum.enzyme.tx import EnzymeTransactionBuilder
from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.statistics.core import calculate_statistics
from tradeexecutor.strategy.asset import get_relevant_assets, build_expected_asset_map
from tradeexecutor.strategy.account_correction import calculate_account_corrections, AccountingCorrectionCause, correct_accounts, UnknownTokenPositionFix
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.testing.ethereumtrader_uniswap_v2 import UniswapV2TestTrader


@pytest.fixture
def hot_wallet(web3, deployer, user_1, usdc: Contract) -> HotWallet:
    """Create hot wallet for the signing tests.

    Top is up with some gas money and 500 USDC.
    """
    private_key = HexBytes(secrets.token_bytes(32))
    account = Account.from_key(private_key)
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    tx_hash = web3.eth.send_transaction({"to": wallet.address, "from": user_1, "value": 15 * 10**18})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = usdc.functions.transfer(wallet.address, 500 * 10**6).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)

    web3.middleware_onion.add(construct_sign_and_send_raw_middleware_anvil(account))

    return wallet


@pytest.fixture()
def routing_model(
        uniswap_v2,
        usdc_asset,
        weth_asset,
        weth_usdc_trading_pair) -> UniswapV2Routing:

    # Allowed exchanges as factory -> router pairs
    factory_router_map = {
        uniswap_v2.factory.address: (uniswap_v2.router.address, uniswap_v2.init_code_hash),
    }

    # Three way ETH quoted trades are routed thru WETH/USDC pool
    allowed_intermediary_pairs = {
        weth_asset.address: weth_usdc_trading_pair.pool_address
    }

    return UniswapV2Routing(
        factory_router_map,
        allowed_intermediary_pairs,
        reserve_token_address=usdc_asset.address,
    )


@pytest.fixture()
def pricing_model(
        web3,
        uniswap_v2,
        pair_universe: PandasPairUniverse,
        routing_model) -> UniswapV2LivePricing:

    pricing_model = UniswapV2LivePricing(
        web3,
        pair_universe,
        routing_model,
    )
    return pricing_model


@flaky.flaky()
def test_enzyme_no_accounting_errors(
    web3: Web3,
    deployer: HexAddress,
    vault: Vault,
    usdc: Contract,
    weth: Contract,
    usdc_asset: AssetIdentifier,
    weth_asset: AssetIdentifier,
    user_1: HexAddress,
    uniswap_v2: UniswapV2Deployment,
    weth_usdc_trading_pair: TradingPairIdentifier,
    pair_universe: PandasPairUniverse,
    hot_wallet: HotWallet,
):
    """Check we get no accounting errors if there are one

    - Open one trading position

    - Accounting errors check should give us a clean pass
    """

    reorg_mon = create_reorganisation_monitor(web3)

    tx_hash = vault.vault.functions.addAssetManagers([hot_wallet.address]).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    sync_model = EnzymeVaultSyncModel(
        web3,
        vault.address,
        reorg_mon,
    )

    state = State()
    sync_model.sync_initial(state)

    # Make two deposits from separate parties
    usdc.functions.transfer(user_1, 500 * 10**6).transact({"from": deployer})
    usdc.functions.approve(vault.comptroller.address, 500 * 10**6).transact({"from": user_1})
    tx_hash = vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Strategy has its reserve balances updated
    sync_model.sync_treasury(datetime.datetime.utcnow(), state)
    assert state.portfolio.get_total_equity() == pytest.approx(500)

    tx_builder = EnzymeTransactionBuilder(hot_wallet, vault)

    # Check we have balance
    assert usdc.functions.balanceOf(tx_builder.get_erc_20_balance_address()).call() == 500 * 10**6

    # Now make a trade
    trader = UniswapV2TestTrader(
        uniswap_v2,
        state=state,
        pair_universe=pair_universe,
        tx_builder=tx_builder,
    )

    position, trade = trader.buy(
        weth_usdc_trading_pair,
        Decimal(500),
        execute=False,
        slippage_tolerance=0.01,
    )

    trader.execute_trades_simple(trader.create_routing_model(), [trade], broadcast=True)

    assert weth.functions.balanceOf(vault.vault.address).call() > 0

    assets = get_relevant_assets(pair_universe, state.portfolio.get_reserve_assets(), state)
    balances = list(sync_model.fetch_onchain_balances(assets))

    # Should be all-in to ETH with some dust error
    assert len(balances) == 2

    # USDC 0.000001 is under DUST_EPSILON limit
    b = balances[0]
    assert b.asset == usdc_asset
    assert b.amount == pytest.approx(Decimal("0.000001"))

    b = balances[1]
    assert b.asset == weth_asset
    assert b.amount == pytest.approx(Decimal("0.310787860635789571"))

    # No corrections needed, balances are correct
    corrections = list(calculate_account_corrections(
        pair_universe,
        state.portfolio.get_reserve_assets(),
        state,
        sync_model))

    assert len(corrections) == 0


@flaky.flaky()
def test_enzyme_correct_accounting_errors(
    web3: Web3,
    deployer: HexAddress,
    vault: Vault,
    usdc: Contract,
    weth: Contract,
    usdc_asset: AssetIdentifier,
    weth_asset: AssetIdentifier,
    user_1: HexAddress,
    user_2: HexAddress,
    uniswap_v2: UniswapV2Deployment,
    weth_usdc_trading_pair: TradingPairIdentifier,
    pair_universe: PandasPairUniverse,
    hot_wallet: HotWallet,
):
    """Correct accounting errors

    - Open one trading position

    - Do behing-the-scenes transfers that break the sync between in the state (internal ledger) and chain

    - Apply the correction to internal ledger accounting

    - Recheck accounting errors disappear
    """

    reorg_mon = create_reorganisation_monitor(web3)

    tx_hash = vault.vault.functions.addAssetManagers([hot_wallet.address]).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    sync_model = EnzymeVaultSyncModel(
        web3,
        vault.address,
        reorg_mon,
    )

    state = State()
    sync_model.sync_initial(state)

    # Make two deposits from separate parties
    usdc.functions.transfer(user_1, 1000 * 10**6).transact({"from": deployer})
    usdc.functions.approve(vault.comptroller.address, 500 * 10**6).transact({"from": user_1})
    tx_hash = vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Strategy has its reserve balances updated
    sync_model.sync_treasury(datetime.datetime.utcnow(), state)
    assert state.portfolio.get_total_equity() == pytest.approx(500)

    reserve_position: ReservePosition = state.portfolio.get_default_reserve_position()
    assert reserve_position.get_quantity() == pytest.approx(Decimal(500))

    tx_builder = EnzymeTransactionBuilder(hot_wallet, vault)

    # Check we have balance
    assert usdc.functions.balanceOf(tx_builder.get_erc_20_balance_address()).call() == 500 * 10**6

    # Now make a trade
    trader = UniswapV2TestTrader(
        uniswap_v2,
        state=state,
        pair_universe=pair_universe,
        tx_builder=tx_builder,
    )

    position, trade = trader.buy(
        weth_usdc_trading_pair,
        Decimal(500),
        execute=False,
        slippage_tolerance=0.01,
    )

    trader.execute_trades_simple(trader.create_routing_model(), [trade], broadcast=True)

    reserve_position: ReservePosition = state.portfolio.get_default_reserve_position()
    assert reserve_position.get_quantity() == pytest.approx(Decimal(0))

    assert weth.functions.balanceOf(vault.vault.address).call() > 0

    #
    # Mess up accounting by not correctly syncing it
    #

    # 1. Add 500 USDC in reserves
    usdc.functions.approve(vault.comptroller.address, 400 * 10**6).transact({"from": user_1})
    tx_hash = vault.comptroller.functions.buyShares(400 * 10**6, 1).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    assert usdc.functions.balanceOf(vault.vault.address).call() == Decimal("400.000001") * Decimal(10**6)

    # 2. Move away tokens from the vault without selling them
    prepared_tx = prepare_transfer(
        vault.deployment,
        vault,
        vault.generic_adapter,
        weth,
        user_2,
        int(0.1 * 10**18),
    )
    tx_hash = prepared_tx.transact({"from": hot_wallet.address})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Now both USDC and WETH balances should be out of sync
    corrections = list(calculate_account_corrections(
        pair_universe,
        state.portfolio.get_reserve_assets(),
        state,
        sync_model))

    assert len(corrections) == 2

    # USDC correction
    reserve_correction = corrections[0]
    assert reserve_correction.type == AccountingCorrectionCause.unknown_cause
    assert isinstance(reserve_correction.position, ReservePosition)
    assert reserve_correction.expected_amount == pytest.approx(Decimal('2.35431888385E-14'))
    assert reserve_correction.actual_amount == pytest.approx(Decimal('400.000001'))
    assert reserve_correction.block_number > 0
    assert isinstance(reserve_correction.timestamp, datetime.datetime)

    # WETH correction
    position_correction = corrections[1]
    assert position_correction.type == AccountingCorrectionCause.unknown_cause
    assert isinstance(position_correction.position, TradingPosition)
    assert position_correction.expected_amount == pytest.approx(Decimal('0.310787860635789571'))
    assert position_correction.actual_amount == pytest.approx(Decimal('0.2107878606357895711'))

    #
    # Correct state (internal ledger)
    #

    vault = sync_model.vault
    tx_builder = EnzymeTransactionBuilder(hot_wallet, vault)

    balance_updates = correct_accounts(
        state,
        corrections,
        strategy_cycle_included_at=None,
        tx_builder=tx_builder,
        interactive=False,
    )
    balance_updates = list(balance_updates)
    assert len(balance_updates) == 2

    # See the balance updates look good
    reserve_update = balance_updates[0]
    assert reserve_update.usd_value == pytest.approx(400)
    assert reserve_update.position_type == BalanceUpdatePositionType.reserve
    assert reserve_update.block_number > 0
    assert reserve_update.quantity == pytest.approx(Decimal('400.0000009999999764568111615'))
    assert reserve_update.strategy_cycle_included_at is None
    assert reserve_update.cause == BalanceUpdateCause.correction

    position_update = balance_updates[1]
    assert position_update.position_type == BalanceUpdatePositionType.open_position
    assert position_update.usd_value == pytest.approx(-160.881444332999)
    assert position_update.block_number > 0
    assert position_update.quantity == Decimal('-0.100000000000000000')
    assert position_update.strategy_cycle_included_at is None
    assert position_update.cause == BalanceUpdateCause.correction

    # Check we updated acconting correction metadata
    assert state.sync.accounting.last_updated_at is not None
    assert state.sync.accounting.last_block_scanned > 0
    assert len(state.sync.accounting.balance_update_refs) == 2

    ref = state.sync.accounting.balance_update_refs[0]
    assert ref.balance_event_id == 2
    assert ref.strategy_cycle_included_at is None
    assert ref.cause == BalanceUpdateCause.correction
    assert ref.position_type == BalanceUpdatePositionType.reserve
    assert ref.usd_value == pytest.approx(400)

    ref = state.sync.accounting.balance_update_refs[1]
    assert ref.balance_event_id == 3
    assert ref.strategy_cycle_included_at is None
    assert ref.cause == BalanceUpdateCause.correction
    assert ref.position_type == BalanceUpdatePositionType.open_position
    assert ref.usd_value == pytest.approx(-160.881444332999)

    # Check that actual position amounts are now correct
    # and match actual amounts
    reserve_position: ReservePosition = state.portfolio.get_default_reserve_position()
    assert len(reserve_position.balance_updates) == 2
    assert reserve_position.get_quantity() == pytest.approx(Decimal('400.000001'))

    position: TradingPosition = state.portfolio.open_positions[1]
    assert len(position.balance_updates) == 1
    assert position.get_quantity() == pytest.approx(Decimal('0.2107878606357895711'))

    #
    # Check state serialises afterwards
    #
    text = state.to_json_safe()
    state2 = State.read_json_blob(text)
    assert not state2.is_empty()

    #
    # No further accounting errors
    #
    further_corrections = list(calculate_account_corrections(
        pair_universe,
        state.portfolio.get_reserve_assets(),
        state,
        sync_model))

    assert len(further_corrections) == 0


def test_enzyme_correct_accounting_no_open_position(
    web3: Web3,
    deployer: HexAddress,
    vault: Vault,
    usdc: Contract,
    weth: Contract,
    usdc_asset: AssetIdentifier,
    weth_asset: AssetIdentifier,
    user_1: HexAddress,
    user_2: HexAddress,
    uniswap_v2: UniswapV2Deployment,
    weth_usdc_trading_pair: TradingPairIdentifier,
    pair_universe: PandasPairUniverse,
    hot_wallet: HotWallet,
):
    """Correct accounting errors

    - Send it WETH that belongs to WETH-USDC position

    - Accounting correction maps the unknow weth to a position

    - Apply the correction to internal ledger accounting

    - Recheck accounting errors disappear
    """
    assert vault.generic_adapter

    reorg_mon = create_reorganisation_monitor(web3)

    tx_hash = vault.vault.functions.addAssetManagers([hot_wallet.address]).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    sync_model = EnzymeVaultSyncModel(
        web3,
        vault.address,
        reorg_mon,
        generic_adapter_address=vault.generic_adapter.address,
    )

    state = State()
    sync_model.sync_initial(state)

    # Make two deposits from separate parties
    usdc.functions.transfer(user_1, 1000 * 10**6).transact({"from": deployer})
    usdc.functions.approve(vault.comptroller.address, 500 * 10**6).transact({"from": user_1})
    tx_hash = vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": user_1})

    assert_transaction_success_with_explanation(web3, tx_hash)

    # Strategy has its reserve balances updated
    sync_model.sync_treasury(datetime.datetime.utcnow(), state)
    assert state.portfolio.get_total_equity() == pytest.approx(500)

    # Send in some WETH
    weth.functions.transfer(vault.vault.address, 2*10**18).transact({"from": deployer})

    asset_to_position_mapping = build_expected_asset_map(state.portfolio, pair_universe=pair_universe)
    assert len(asset_to_position_mapping) == 2   # USDC, WETH

    # Now the strategy WETH balances should be out of sync
    corrections = list(calculate_account_corrections(
        pair_universe,
        state.portfolio.get_reserve_assets(),
        state,
        sync_model))

    # Unknown WETH
    assert len(corrections) == 1

    # WETH correction
    position_correction = corrections[0]
    assert position_correction.type == AccountingCorrectionCause.unknown_cause
    assert position_correction.asset == weth_asset
    assert position_correction.position is None
    assert position_correction.expected_amount == pytest.approx(Decimal('0'))
    assert position_correction.actual_amount == pytest.approx(Decimal('2'))

    #
    # Correct state (internal ledger)
    #

    vault = sync_model.vault
    tx_builder = EnzymeTransactionBuilder(hot_wallet, vault)
    balance_updates = correct_accounts(
        state,
        corrections,
        strategy_cycle_included_at=None,
        interactive=False,
        tx_builder=tx_builder,
        unknown_token_receiver=user_1,
        token_fix_method=UnknownTokenPositionFix.transfer_away,
    )
    balance_updates = list(balance_updates)
    assert len(balance_updates) == 0

    assert weth.functions.balanceOf(user_1).call() == pytest.approx(Decimal(2*10**18))

    #
    # No further accounting errors
    #
    further_corrections = list(calculate_account_corrections(
        pair_universe,
        state.portfolio.get_reserve_assets(),
        state,
        sync_model))

    assert len(further_corrections) == 0


def test_correct_accounting_errors_for_zero_position(
    web3: Web3,
    deployer: HexAddress,
    vault: Vault,
    usdc: Contract,
    weth: Contract,
    usdc_asset: AssetIdentifier,
    weth_asset: AssetIdentifier,
    user_1: HexAddress,
    user_2: HexAddress,
    uniswap_v2: UniswapV2Deployment,
    weth_usdc_trading_pair: TradingPairIdentifier,
    pair_universe: PandasPairUniverse,
    hot_wallet: HotWallet,
):
    """Correct accounting errors on a position which on-chain balance has gone zero.

    - Position has lost all its tokens
    """

    reorg_mon = create_reorganisation_monitor(web3)

    tx_hash = vault.vault.functions.addAssetManagers([hot_wallet.address]).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    sync_model = EnzymeVaultSyncModel(
        web3,
        vault.address,
        reorg_mon,
    )

    state = State()
    sync_model.sync_initial(state)

    # Make two deposits from separate parties
    usdc.functions.transfer(user_1, 1000 * 10**6).transact({"from": deployer})
    usdc.functions.approve(vault.comptroller.address, 500 * 10**6).transact({"from": user_1})
    tx_hash = vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": user_1})

    assert_transaction_success_with_explanation(web3, tx_hash)

    # Strategy has its reserve balances updated
    sync_model.sync_treasury(datetime.datetime.utcnow(), state)
    assert state.portfolio.get_total_equity() == pytest.approx(500)

    reserve_position: ReservePosition = state.portfolio.get_default_reserve_position()
    assert reserve_position.get_quantity() == pytest.approx(Decimal(500))

    tx_builder = EnzymeTransactionBuilder(hot_wallet, vault)

    # Check we have balance
    assert usdc.functions.balanceOf(tx_builder.get_erc_20_balance_address()).call() == 500 * 10**6

    # Now make a trade
    trader = UniswapV2TestTrader(
        uniswap_v2,
        state=state,
        pair_universe=pair_universe,
        tx_builder=tx_builder,
    )

    position, trade = trader.buy(
        weth_usdc_trading_pair,
        Decimal(500),
        execute=False,
        slippage_tolerance=0.01,
    )

    trader.execute_trades_simple(trader.create_routing_model(), [trade], broadcast=True)

    reserve_position: ReservePosition = state.portfolio.get_default_reserve_position()
    assert reserve_position.get_quantity() == pytest.approx(Decimal(0))

    on_chain_balance = weth.functions.balanceOf(vault.vault.address).call()
    assert weth.functions.balanceOf(vault.vault.address).call() > 0

    #
    # Mess up accounting by not correctly syncing it
    #

    # Move away tokens from the vault without selling them
    prepared_tx = prepare_transfer(
        vault.deployment,
        vault,
        vault.generic_adapter,
        weth,
        user_2,
        on_chain_balance,
    )
    tx_hash = prepared_tx.transact({"from": hot_wallet.address})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Now both USDC and WETH balances should be out of sync
    corrections = list(calculate_account_corrections(
        pair_universe,
        state.portfolio.get_reserve_assets(),
        state,
        sync_model))

    assert len(corrections) == 1

    # WETH correction 0.31 -> 0
    position_correction = corrections[0]
    assert position_correction.type == AccountingCorrectionCause.unknown_cause
    assert isinstance(position_correction.position, TradingPosition)
    assert position_correction.expected_amount == pytest.approx(Decimal('0.310787860635789571'))
    assert position_correction.actual_amount == pytest.approx(Decimal('0'))

    #
    # Correct state (internal ledger)
    #

    vault = sync_model.vault
    tx_builder = EnzymeTransactionBuilder(hot_wallet, vault)

    balance_updates = correct_accounts(
        state,
        corrections,
        strategy_cycle_included_at=None,
        tx_builder=tx_builder,
        interactive=False,
    )
    balance_updates = list(balance_updates)
    assert len(balance_updates) == 1

    #
    # No further accounting errors
    #
    further_corrections = list(calculate_account_corrections(
        pair_universe,
        state.portfolio.get_reserve_assets(),
        state,
        sync_model))
    assert len(further_corrections) == 0

    # Position has been closed in the portfolio
    portfolio = state.portfolio
    assert len(portfolio.open_positions) == 0
    assert len(portfolio.frozen_positions) == 0
    assert len(portfolio.closed_positions) == 1

    # Check the repaired position behaves correctly
    position = portfolio.closed_positions[1]
    assert len(position.trades) == 2
    position.is_closed()
    assert position.get_quantity() == 0
    assert position.get_value() == 0
    assert position.is_reduced()
    assert position.get_realised_profit_usd() == 0.0

    # See that portfolio statistics calculations do not get screwed over because of
    # manual repair entriesa
    stats = calculate_statistics(
        datetime.datetime.now(),
        portfolio,
        ExecutionMode.unit_testing_trading
    )

    assert stats is not None

    #
    # Check state serialises afterwards
    #
    text = state.to_json_safe()
    state2 = State.read_json_blob(text)
    assert not state2.is_empty()
