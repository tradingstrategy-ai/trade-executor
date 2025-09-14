"""Test depositing and redeeming ERC-4626 vaults as a trade."""

import datetime
import os
from decimal import Decimal

import pytest

from eth_typing import HexAddress
from web3 import Web3

from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.lagoon.deposit_redeem import ERC7540DepositTicket, ERC7540RedemptionTicket
from eth_defi.lagoon.testing import force_lagoon_settle
from eth_defi.lagoon.vault import LagoonVault
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.ethereum.vault.staged_deposit_redeem import MultiStageDepositRedeemManager, get_multi_stage_state, start_multi_stage_deposit, can_complete_multi_stage, finish_multi_stage_trade

from tradeexecutor.ethereum.vault.vault_routing import VaultRouting
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import MultiStageTradeKind
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")
pytestmark = pytest.mark.skipif(not JSON_RPC_BASE, reason="No JSON_RPC_BASE environment variable")


@pytest.fixture()
def vault(web3) -> LagoonVault:
    # https://app.lagoon.finance/vault/8453/0xb09f761cb13baca8ec087ac476647361b6314f98
    vault = create_vault_instance(web3, "0xb09f761cb13baca8ec087ac476647361b6314f98", features={ERC4626Feature.lagoon_like, ERC4626Feature.erc_7540_like})
    assert vault.features
    return vault


def test_vault_routing(
    vault: LagoonVault,
    routing_model: GenericRouting,
    strategy_universe: TradingStrategyUniverse,
):
    """Check we know how to route vault trades."""

    pair = strategy_universe.get_pair_by_smart_contract(vault.address)
    assert pair.is_vault()
    assert pair.get_vault_features() == {ERC4626Feature.lagoon_like}
    assert pair.get_vault_protocol() == "lagoon"
    assert vault.name == "IPOR Stablecoin Vault"

    routing_id = routing_model.pair_configurator.match_router(pair)
    protocol_config = routing_model.pair_configurator.get_config(routing_id)
    assert protocol_config.routing_id.router_name == "vault"
    assert protocol_config.routing_id.exchange_slug is None
    assert isinstance(protocol_config.routing_model, VaultRouting)


def test_erc_7540_deposit(
    web3: Web3,
    vault: LagoonVault,
    strategy_universe,
    execution_model,
    routing_model: GenericRouting,
    pricing_model,
    sync_model: HotWalletSyncModel,
    base_usdc: AssetIdentifier,
    lagoon_722_capital_manager: HexAddress,
):
    """Do a deposit to Lagoon vault and perform Uniswap v2 token buy (three legs)."""

    # Forked vault
    assert vault.name == "722Capital-USDC"

    state = State()
    pair = strategy_universe.get_pair_by_smart_contract(vault.address)
    assert pair.is_erc_7540()

    sync_model.sync_initial(
        state,
        reserve_asset=base_usdc,
        reserve_token_price=1.0,
    )

    position_manager = PositionManager(
        datetime.datetime.utcnow(),
        universe=strategy_universe,
        state=state,
        pricing_model=pricing_model,
        default_slippage_tolerance=0.20,
    )

    #
    # 1. Create the first half of the deposit
    #

    assert pair.is_multi_stage_deposit()

    t = start_multi_stage_deposit(
        position_manager,
        pair,
        amount=500
    )

    trades = [t]
    assert t.is_planned()
    assert "multi_stage_state" in t.other_data
    assert t.is_multi_stage()

    routing_state_details = execution_model.get_routing_state_details()
    routing_state = routing_model.create_routing_state(strategy_universe, routing_state_details)

    execution_model.initialize()

    execution_model.execute_trades(
        datetime.datetime.utcnow(),
        state,
        trades,
        routing_model,
        routing_state,
        check_balances=True,
    )

    # Part 1 of the deposit done
    assert t.is_executed(), f"Trade did not execute: {t}: {t.get_revert_reason()}"
    assert t.is_success(), f"Trade failed: {t.get_revert_reason()}"
    assert t.is_buy()
    assert t.is_multi_stage()
    assert t.executed_price == pytest.approx(1.0)
    assert t.executed_quantity == pytest.approx(Decimal(500))
    assert t.executed_reserve == 500
    assert t.get_multi_stage_kind() == MultiStageTradeKind.deposit_start
    multi_stage_state = get_multi_stage_state(t)
    assert isinstance(multi_stage_state.deposit_ticket, ERC7540DepositTicket)

    # The position is considered open even when we do not have shares yet
    position = position_manager.get_current_position_for_pair(pair)
    assert position.get_quantity() == pytest.approx(Decimal(500))

    assert position.is_multi_stage()
    assert position.is_multi_stage_in_process()

    #
    # 2. Complete the second half of deposit
    #

    # Need to be settled by Lagoon vault manager
    assert not can_complete_multi_stage(web3, position)
    force_lagoon_settle(
        vault,
        lagoon_722_capital_manager,
    )
    assert can_complete_multi_stage(web3, position)

    t = finish_multi_stage_trade(
        position_manager,
        position,
    )

    assert t.is_planned()
    assert "multi_stage_state" in t.other_data
    assert t.is_multi_stage()
    assert t.get_multi_stage_kind() == MultiStageTradeKind.deposit_finish

    execution_model.execute_trades(
        datetime.datetime.utcnow(),
        state,
        [t],
        routing_model,
        routing_state,
        check_balances=True,
    )

    # Part 2 of the deposit done
    assert t.is_executed(), f"Trade did not execute: {t}: {t.get_revert_reason()}"
    assert t.is_success(), f"Trade failed: {t.get_revert_reason()}"
    assert t.is_buy()
    assert t.is_multi_stage()
    assert t.executed_price == 0
    assert t.executed_quantity == pytest.approx(Decimal(500))
    assert t.executed_reserve == 500
    assert t.get_multi_stage_kind() == MultiStageTradeKind.deposit_finish
    multi_stage_state = get_multi_stage_state(t)
    assert multi_stage_state.deposit_ticket in None

    # Then redeem shares back
    trades = position_manager.close_all()
    assert len(trades) == 1
    t = trades[0]

    execution_model.execute_trades(
        datetime.datetime.utcnow(),
        state,
        trades,
        routing_model,
        routing_state,
        check_balances=True,
    )

    assert t.is_success(), f"Trade failed: {t.blockchain_transactions[0].revert_reason}"
    assert t.is_sell()
    assert t.planned_quantity == pytest.approx(Decimal(-9.67523177))
    assert t.executed_price == pytest.approx(1.0335668671701836)
    assert t.executed_quantity == pytest.approx(Decimal(-9.67523178))
    assert t.executed_reserve == pytest.approx(Decimal('9.999999'))

    assert position.is_closed()
