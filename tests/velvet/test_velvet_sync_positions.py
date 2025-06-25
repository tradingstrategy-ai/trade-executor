"""Test Velvet sync model.

- Handle special Velvet deposit/redemption model
"""
import datetime
import os
from decimal import Decimal

import flaky
import pytest
from eth_typing import HexAddress
from web3 import Web3

from eth_defi.token import TokenDetails
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.velvet import VelvetVault

from tradeexecutor.ethereum.velvet.vault import VelvetVaultSyncModel
from tradeexecutor.state.balance_update import BalanceUpdateCause
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.state import State
from tradeexecutor.strategy.asset import build_expected_asset_map
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair, TradingStrategyUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.pair import PandasPairUniverse

#: Detect Github Actions
CI = os.environ.get("CI", None) is not None

pytestmark = pytest.mark.skipif(CI, reason="This is broken most of the time, so there is no need to try to maintain it")


@pytest.fixture()
def deposit_user() -> HexAddress:
    """A user that has preapproved 5 USDC deposit for the vault above, no approve() needed."""
    return "0x9C5749f73e3D8728DDC77d69b3DB3C60B91A91E2"


def test_check_velvet_universe(
        velvet_test_vault_pair_universe: PandasPairUniverse,
):
    """Check we have good universe to map pairs to positions."""
    pair_universe = velvet_test_vault_pair_universe

    assert pair_universe.get_count() == 5

    for pair in pair_universe.iterate_pairs():
        assert pair.base_token_symbol, f"Got {pair}"
        assert pair.quote_token_symbol, f"Got {pair}"


def test_check_price_on_base_uniswap_v3(
        velvet_pricing_model: PricingModel,
        velvet_test_vault_strategy_universe: PandasPairUniverse,
):
    """Check that we have a good price feed."""

    strategy_universe = velvet_test_vault_strategy_universe
    pricing_model = velvet_pricing_model

    pair_desc = (ChainId.base, "uniswap-v3", "DogInMe", "WETH", 0.01)

    pair = strategy_universe.get_pair_by_human_description(pair_desc)

    # Read DogMeIn/USDC pool price
    # Note this pool lacks liquidity, so we do not care about the value
    mid_price = pricing_model.get_mid_price(
        datetime.datetime.utcnow(),
        pair,
    )

    assert mid_price > 0


def test_velvet_fetch_balances(
        base_example_vault: VelvetVault,
        velvet_test_vault_pair_universe: PandasPairUniverse,
):
    """Check we fetch balances for onchain positions correctly.

    - Read DogInMe balance onchain
    """

    pair_universe = velvet_test_vault_pair_universe

    sync_model = VelvetVaultSyncModel(
        vault=base_example_vault,
        hot_wallet=None,
    )

    asset_to_position_map = build_expected_asset_map(
        Portfolio(),
        pair_universe=pair_universe,
        ignore_reserve=True,
    )

    pair_desc = (ChainId.base, "uniswap-v3", "DogInMe", "WETH", 0.01)
    pair = translate_trading_pair(
        pair_universe.get_pair_by_human_description(pair_desc),
    )
    dog_in_me = pair.base

    balances = sync_model.fetch_onchain_balances(
        list(asset_to_position_map.keys())
    )

    balance_map = {a.asset: a for a in balances}

    onchain_balance_data = balance_map[dog_in_me]
    assert onchain_balance_data.amount > 0


def test_velvet_sync_positions_initial(
        base_example_vault: VelvetVault,
        base_usdc: AssetIdentifier,
        velvet_test_vault_strategy_universe: TradingStrategyUniverse,
        velvet_pricing_model: PricingModel,
):
    """Sync velvet open positions

    - Do initial deposit scan.

    - Capture the initial USDC in the vault as a treasury

    - Capture DogMeIn open position
    """

    strategy_universe = velvet_test_vault_strategy_universe
    pair_universe = strategy_universe.data_universe.pairs
    pricing_model = velvet_pricing_model
    assert pair_universe.get_count() == 5

    sync_model = VelvetVaultSyncModel(
        vault=base_example_vault,
        hot_wallet=None,
    )

    state = State()
    portfolio = state.portfolio

    # Sync USDC
    sync_model.sync_initial(
        state,
        reserve_asset=base_usdc,
        reserve_token_price=1.0,
    )
    cycle = datetime.datetime.utcnow()
    sync_model.sync_treasury(cycle, state)
    assert portfolio.get_cash() == pytest.approx(2.674828)

    # Sync DogInMe - creates initial position no event generated
    events = sync_model.sync_positions(
        datetime.datetime.utcnow(),
        state,
        strategy_universe=strategy_universe,
        pricing_model=pricing_model,
    )
    assert len(events) == 0

    # We have DogInMe position
    assert len(portfolio.open_positions) == 1
    dog_in_me_pos = portfolio.open_positions[1]
    assert dog_in_me_pos.get_value() > 0


@pytest.mark.skipif(CI, reason="Skipped on continuous integration due to Velvet/Enso stability issues")
def test_velvet_sync_positions_deposit(
        web3: Web3,
        base_example_vault: VelvetVault,
        base_usdc: AssetIdentifier,
        base_doginme: AssetIdentifier,
        velvet_test_vault_strategy_universe: TradingStrategyUniverse,
        velvet_pricing_model: PricingModel,
        base_usdc_token: TokenDetails,
        deposit_user: HexAddress,
):
    """Sync velvet open positions

    - Do initial deposit scan.

    - Capture the initial USDC in the vault

    - Velvet distributes USDC to all open positions

    - See all positions get the increase
    """

    vault = base_example_vault
    strategy_universe = velvet_test_vault_strategy_universe
    pair_universe = strategy_universe.data_universe.pairs
    pricing_model = velvet_pricing_model
    usdc_contract = base_usdc_token.contract

    sync_model = VelvetVaultSyncModel(
        vault=base_example_vault,
        hot_wallet=None,
    )

    state = State()
    portfolio = state.portfolio

    # Get the initial positions
    sync_model.sync_initial(
        state,
        reserve_asset=base_usdc,
        reserve_token_price=1.0,
    )
    sync_model.sync_treasury(
        datetime.datetime.utcnow(),
        state,
        supported_reserves=state.portfolio.get_reserve_assets(),
    )
    sync_model.sync_positions(
        datetime.datetime.utcnow(),
        state,
        strategy_universe=strategy_universe,
        pricing_model=pricing_model,
    )

    # We have DogInMe position + cash
    assert len(portfolio.open_positions) == 1
    assert len(state.portfolio.get_reserve_assets()) == 1
    assert state.portfolio.get_cash() == pytest.approx(2.674828)
    assert portfolio.open_positions[1].get_quantity() == pytest.approx(Decimal(580.917745152826802993))

    # Do velvet deposit

    # Velvet deposit manager on Base,
    # the destination of allowance
    deposit_manager = "0xe4e23120a38c4348D7e22Ab23976Fa0c4Bf6e2ED"

    # Check there is ready-made manual approve() waiting onchain
    allowance = usdc_contract.functions.allowance(
        Web3.to_checksum_address(deposit_user),
        Web3.to_checksum_address(deposit_manager),
    ).call()
    amount = 4999999
    assert allowance == amount

    # Prepare the deposit tx payload
    tx_data = vault.prepare_deposit_with_enso(
        from_=deposit_user,
        deposit_token_address=usdc_contract.address,
        amount=amount,
        slippage=0.20,
    )
    assert tx_data["gas"] > 1_000_000
    tx_hash = web3.eth.send_transaction(tx_data)
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Velvet will split 5 USDC between the open DogInMe position
    # and the cash position

    # Check USDC update
    events = sync_model.sync_treasury(
        datetime.datetime.utcnow(),
        state,
        supported_reserves=state.portfolio.get_reserve_assets(),
    )
    assert len(events) == 1
    evt = events[0]
    assert evt.balance_update_id == 2
    assert evt.chain_id == 8453
    assert evt.previous_update_at is None
    assert evt.quantity > 2
    assert evt.tx_hash is None  # poll mode does not get txs
    assert evt.cause == BalanceUpdateCause.deposit
    assert evt.asset == base_usdc
    assert state.portfolio.get_cash() > 3

    # Check DogInMe update,
    # quantity has increased
    events = sync_model.sync_positions(
        datetime.datetime.utcnow(),
        state,
        strategy_universe=strategy_universe,
        pricing_model=pricing_model,
    )
    assert len(events) == 1
    evt = events[0]
    assert evt.balance_update_id == 3
    assert evt.chain_id == 8453
    assert evt.old_balance == pytest.approx(Decimal(580.917745152826802993))
    assert evt.quantity > 100  # Depends on market conditions
    assert 0 < evt.usd_value < 1
    assert evt.cause == BalanceUpdateCause.vault_flow
    assert evt.asset == base_doginme
    assert state.portfolio.open_positions[1].get_quantity() > 600


# :test_velvet_sync_positions_redeem - requests.exceptions.ReadTimeout: HTTPConnectionPool(host='localhost', port=27100): Read timed out. (read timeout=60)
@pytest.mark.skipif(CI, reason="Skipped on continuous integration due to Velvet/Enso stability issues")
@flaky.flaky()
def test_velvet_sync_positions_redeem(
        web3: Web3,
        base_example_vault: VelvetVault,
        base_usdc: AssetIdentifier,
        base_doginme: AssetIdentifier,
        velvet_test_vault_strategy_universe: TradingStrategyUniverse,
        velvet_pricing_model: PricingModel,
        base_usdc_token: TokenDetails,
        existing_shareholder: HexAddress,
):
    """Sync velvet open positions.

    - Do initial deposit scan.

    - Capture the initial USDC and DogInMe in the vault

    - Redeem all shares using an account with pre-set approve()
    """

    vault = base_example_vault
    strategy_universe = velvet_test_vault_strategy_universe
    pair_universe = strategy_universe.data_universe.pairs
    pricing_model = velvet_pricing_model
    usdc_contract = base_usdc_token.contract

    sync_model = VelvetVaultSyncModel(
        vault=base_example_vault,
        hot_wallet=None,
    )

    state = State()
    portfolio = state.portfolio

    # Get the initial positions
    sync_model.sync_initial(
        state,
        reserve_asset=base_usdc,
        reserve_token_price=1.0,
    )
    sync_model.sync_treasury(
        datetime.datetime.utcnow(),
        state,
        supported_reserves=state.portfolio.get_reserve_assets(),
    )
    sync_model.sync_positions(
        datetime.datetime.utcnow(),
        state,
        strategy_universe=strategy_universe,
        pricing_model=pricing_model,
    )

    # We have DogInMe position + cash
    assert len(portfolio.open_positions) == 1
    assert len(state.portfolio.get_reserve_assets()) == 1
    assert state.portfolio.get_cash() == pytest.approx(2.674828)
    assert portfolio.open_positions[1].get_quantity() == pytest.approx(Decimal(580.917745152826802993))

    # Do Velvet redemption
    # Redeem all shares

    withdrawal_manager = "0x99e9C4d3171aFAA3075D0d1aE2Bb42B5E53aEdAB"

    shares = vault.share_token.fetch_balance_of(existing_shareholder)

    tx_data = vault.prepare_redemption(
        from_=existing_shareholder,
        amount=vault.share_token.convert_to_raw(shares),
        withdraw_token_address=base_usdc.address,
        slippage=0.10,
    )
    assert tx_data["to"] == withdrawal_manager
    tx_hash = web3.eth.send_transaction(tx_data)
    assert_transaction_success_with_explanation(web3, tx_hash)

    #
    # Sync positions after a full redemption
    #

    # Check USDC update
    events = sync_model.sync_treasury(
        datetime.datetime.utcnow(),
        state,
        supported_reserves=state.portfolio.get_reserve_assets(),
    )
    assert len(events) == 1
    evt = events[0]
    assert evt.balance_update_id == 2
    assert evt.chain_id == 8453
    assert evt.previous_update_at is None
    assert evt.quantity < 0  # Taking out all USDC
    assert evt.tx_hash is None  # poll mode does not get txs
    assert evt.cause == BalanceUpdateCause.redemption
    assert evt.asset == base_usdc
    assert state.portfolio.get_cash() == pytest.approx(0)

    # Check DogInMe update,
    # quantity has increased
    events = sync_model.sync_positions(
        datetime.datetime.utcnow(),
        state,
        strategy_universe=strategy_universe,
        pricing_model=pricing_model,
    )
    assert len(events) == 1
    evt = events[0]
    assert evt.balance_update_id == 3
    assert evt.chain_id == 8453
    assert evt.old_balance == pytest.approx(Decimal(580.917745152826802993))
    assert evt.quantity < 0  # Depends on market conditions
    assert -1 < evt.usd_value < 0
    assert evt.cause == BalanceUpdateCause.vault_flow
    assert evt.asset == base_doginme
    assert len(state.portfolio.open_positions) == 0
    assert len(state.portfolio.closed_positions) == 1
    assert state.portfolio.closed_positions[1].get_quantity() == pytest.approx(0)
