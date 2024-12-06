"""Test Velvet vault deposits are correctly read."""
import datetime

import pytest
from eth_typing import HexAddress
from web3 import Web3

from eth_defi.token import TokenDetails
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.velvet import VelvetVault

from tradeexecutor.ethereum.velvet.vault import VelvetVaultSyncModel
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.portfolio import ReserveMissing
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.chain import ChainId


def test_velvet_treasury_initialise(
    base_example_vault: VelvetVault,
    base_usdc: AssetIdentifier,
):
    """Initialise Velvet treasury

    - Do initial deposit scan.

    - Capture the initial USDC in the vault as a treasury
    """

    sync_model = VelvetVaultSyncModel(
        vault=base_example_vault,
        hot_wallet=None,
    )

    state = State()
    treasury = state.sync.treasury
    portfolio = state.portfolio
    assert len(portfolio.reserves) == 0
    assert len(treasury.balance_update_refs) == 0
    with pytest.raises(ReserveMissing):
        portfolio.get_default_reserve_position()

    sync_model.sync_initial(
        state,
        reserve_asset=base_usdc,
        reserve_token_price=1.0,
    )
    assert len(portfolio.reserves) == 1  # USDC added as the reserve asset
    assert len(treasury.balance_update_refs) == 0  # No deposits processed yet

    # We have reserve position now, but without any balance
    reserve_position = portfolio.get_default_reserve_position()
    assert len(reserve_position.balance_updates) == 0  # No deposits processed yet
    assert reserve_position.asset.get_identifier() == "8453-0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"

    # Process the initial deposits
    cycle = datetime.datetime.utcnow()
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 1
    assert treasury.last_block_scanned > 0
    assert treasury.last_updated_at is not None
    assert treasury.last_cycle_at is not None
    assert len(treasury.balance_update_refs) == 1
    assert len(reserve_position.balance_updates) == 1

    # We scan again, no changes
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 0
    assert len(treasury.balance_update_refs) == 1
    assert len(reserve_position.balance_updates) == 1

    # Check we account USDC correctly
    assert portfolio.get_cash() == pytest.approx(2.674828)


@pytest.mark.skip(reason="Velvet is broken")
def test_velvet_sync_deposit(
    web3: Web3,
    base_example_vault: VelvetVault,
    base_usdc: AssetIdentifier,
    base_doginme: AssetIdentifier,
    deposit_user,
        base_usdc_address: HexAddress,
    velvet_test_vault_strategy_universe: TradingStrategyUniverse,
    velvet_test_vault_pricing_model: PricingModel,
    base_usdc_token: TokenDetails,
):
    """Sync Velvet deposit"""

    vault = base_example_vault
    strategy_universe = velvet_test_vault_strategy_universe
    pricing_model = velvet_test_vault_pricing_model

    sync_model = VelvetVaultSyncModel(
        vault=vault,
        hot_wallet=None,
    )

    state = State()
    portfolio = state.portfolio

    sync_model.sync_initial(
        state,
        reserve_asset=base_usdc,
        reserve_token_price=1.0,
    )
    assert len(portfolio.reserves) == 1  # USDC added as the reserve asset

    pair = strategy_universe.get_pair_by_human_description(
        (ChainId.base, "uniswap-v3", "DogInMe", "WETH")
    )
    assert pair is not None

    # Process the initial deposits
    cycle = datetime.datetime.utcnow()
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 1
    assert portfolio.get_cash() == pytest.approx(2.674828)

    events = sync_model.sync_positions(cycle, state, strategy_universe, pricing_model)
    assert len(events) == 0  # TODO: There is no event generated yet because we magically open spot position
    dog_in_me_position_size = portfolio.get_position_by_trading_pair(pair).get_value()
    assert dog_in_me_position_size > 0  # Value floats by the day

    #
    # Process additional deposits
    #

    # Velvet deposit manager on Base,
    # the destination of allowance
    deposit_manager = "0xe4e23120a38c4348D7e22Ab23976Fa0c4Bf6e2ED"

    # Check there is ready-made manual approve() waiting onchain
    allowance = base_usdc_token.contract.functions.allowance(
        Web3.to_checksum_address(deposit_user),
        Web3.to_checksum_address(deposit_manager),
        ).call()
    assert allowance == 5 * 10**6

    tx_data = sync_model.vault.prepare_deposit_with_enso(
        from_=deposit_user,
        deposit_token_address=base_usdc_address,
        amount=4 * 10**6,
    )
    tx_hash = web3.eth.send_transaction(tx_data)
    assert_transaction_success_with_explanation(web3, tx_hash)

    #
    # Sync updated balances
    #

    # Process the initial deposits
    cycle = datetime.datetime.utcnow()
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 1
    assert portfolio.get_cash() == pytest.approx(7.023765)

    # DogInMe balance has increased after the deposit
    #
    assert portfolio.get_position_by_trading_pair(pair).get_value() > dog_in_me_position_size

    #
    # Check empty deposit cycle yields no events
    #
    cycle = datetime.datetime.utcnow()
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 0
    assert portfolio.get_cash() == pytest.approx(7.023765)