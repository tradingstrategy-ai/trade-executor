"""Repair a trade that was not broadcasted.

- Uses a live state dump with one failed broadcasted tx
  (tx hash has not been confirmed to be on-chain yet).

- Runs a simulation using a historical archive block on Polygon
"""
import datetime
import os
from _decimal import Decimal
from logging import Logger
from pathlib import Path

import pytest
from tradingstrategy.chain import ChainId
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil

from tradingstrategy.client import Client

from tradeexecutor.cli.bootstrap import create_execution_and_sync_model
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.ethereum.rebroadcast import rebroadcast_all
from tradeexecutor.ethereum.web3config import Web3Config
from tradeexecutor.state.state import State, UncleanState
from tradeexecutor.strategy.approval import UncheckedApprovalModel
from tradeexecutor.strategy.bootstrap import make_factory_from_strategy_mod
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.execution_model import AssetManagementMode
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.strategy.strategy_module import read_strategy_module
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel
from tradeexecutor.strategy.universe_model import UniverseOptions


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture()
def anvil() -> AnvilLaunch:
    """Launch Anvil for the test backend."""

    anvil = launch_anvil(
        fork_url=os.environ["JSON_RPC_POLYGON_ARCHIVE"],
        fork_block_number=49_132_512,
    )
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture
def web3config(anvil: AnvilLaunch) -> Web3Config:
    """Set up a local unit testing blockchain."""
    web3config = Web3Config.setup_from_environment(
        gas_price_method=None,
        json_rpc_polygon=anvil.json_rpc_url,
        unit_testing=True,
    )

    web3config.set_default_chain(ChainId.polygon)
    return web3config


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def state() -> State:
    """A state dump with some failed trades we need to unwind.

    Taken as a snapshot from alpha version trade execution run.
    """
    f = os.path.join(os.path.dirname(__file__), "damaged-broadcast.json")
    return State.from_json(open(f, "rt").read())


@pytest.fixture(scope="module")
def strategy_file() -> Path:
    """A state dump with some failed trades we need to unwind.

    Taken as a snapshot from alpha version trade execution run.
    """
    f = os.path.join(os.path.dirname(__file__), "enzyme-polygon-eth-usdc.py")
    return Path(f)


def test_assess_broadcast_failed(
        state: State,
):
    """We see a trade that has unbroadcasted transactions."""
    with pytest.raises(UncleanState):
        state.check_if_clean()


@pytest.mark.skipif(not os.environ.get("JSON_RPC_POLYGON_ARCHIVE"), reason="Set JSON_RPC_POLYGON_ARCHIVE environment variable to run this test")
@pytest.mark.skipif(not os.environ.get("PRIVATE_KEY"), reason="This special needs related private key, because we do not support Anvil unlocked accounts yet")
def test_broadcast_and_repair_after(
    logger: Logger,
    state: State,
    persistent_test_client: Client,
    web3config: Web3Config,
    strategy_file,
):
    """Broadcast the unbroadcasted txs and repair after.

    - Currently only runnable as local, w/associated private key needed

    TODO: Currently this a slippage disparity between Uniswap v3 and Enzyme that needs to be fixed.
    """

    client = persistent_test_client

    mod = read_strategy_module(strategy_file)

    execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_execution_and_sync_model(
        asset_management_mode=AssetManagementMode.enzyme,
        private_key=os.environ["PRIVATE_KEY"],
        web3config=web3config,
        confirmation_timeout=datetime.timedelta(seconds=30),
        confirmation_block_count=0,
        min_gas_balance=Decimal(0),
        max_slippage=0.005,
        # Taken from the on-chain deployment info
        vault_address="0x6E321256BE0ABd2726A234E8dBFc4d3caf255AE0",
        vault_adapter_address="0x07f7eB451DfeeA0367965646660E85680800E352",
        vault_payment_forwarder_address="0x057bfE6A467e37636462AA92733C04a8D05c3f74",
        routing_hint=mod.trade_routing,
    )

    # Tell the trade execution we are running Anvil
    execution_model.mainnet_fork = True

    execution_context = unit_test_execution_context

    # Set up the strategy engine
    factory = make_factory_from_strategy_mod(mod)
    run_description: StrategyExecutionDescription = factory(
        execution_model=execution_model,
        execution_context=execution_context,
        timed_task_context_manager=execution_context.timed_task_context_manager,
        sync_model=sync_model,
        valuation_model_factory=valuation_model_factory,
        pricing_model_factory=pricing_model_factory,
        approval_model=UncheckedApprovalModel(),
        client=client,
        routing_model=None,
        run_state=RunState(),
    )

    # We construct the trading universe to know what's our reserve asset
    universe_model: TradingStrategyUniverseModel = run_description.universe_model
    universe = universe_model.construct_universe(
        datetime.datetime.utcnow(),
        execution_context.mode,
        UniverseOptions()
    )

    runner = run_description.runner
    routing_model = runner.routing_model
    routing_state, pricing_model, valuation_method = runner.setup_routing(universe)

    # Get the latest nonce from the chain
    sync_model.resync_nonce()

    trades, txs = rebroadcast_all(
        web3config.get_default(),
        state,
        execution_model,
        routing_model,
        routing_state,
    )

    assert len(trades) == 1
    assert len(txs) == 1

    t = trades[0]
    assert t.is_success()

    for tx in t.blockchain_transactions:
        assert tx.is_success()

    #
    # Run second time to ensure there are no further changes
    #

    trades, txs = rebroadcast_all(
        web3config.get_default(),
        state,
        execution_model,
        routing_model,
        routing_state
    )
    assert len(trades) == 0
    assert len(txs) == 0

    state.check_if_clean()