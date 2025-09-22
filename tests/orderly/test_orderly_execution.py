"""Test Orderly execution model."""

import os
import pytest
from web3 import Web3

from tradeexecutor.ethereum.orderly.orderly_vault import OrderlyVault
from tradeexecutor.ethereum.orderly.orderly_execution import OrderlyExecution
from tradeexecutor.ethereum.orderly.tx import OrderlyTransactionBuilder
from tradeexecutor.ethereum.orderly.orderly_routing import OrderlyRouting
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


JSON_RPC_ARBITRUM_SEPOLIA = os.environ.get("JSON_RPC_ARBITRUM_SEPOLIA")
pytestmark = pytest.mark.skipif(not JSON_RPC_ARBITRUM_SEPOLIA, reason="No JSON_RPC_ARBITRUM_SEPOLIA environment variable")



def test_orderly_execution_comprehensive(
    orderly_vault: OrderlyVault,
    orderly_tx_builder: OrderlyTransactionBuilder,
    orderly_strategy_universe: TradingStrategyUniverse,
    broker_id: str,
    orderly_account_id: str,
):
    """Test comprehensive OrderlyExecution functionality including initialization, validation, routing, and state details."""

    # Test initialization
    execution = OrderlyExecution(
        vault=orderly_vault,
        broker_id=broker_id,
        orderly_account_id=orderly_account_id,
        tx_builder=orderly_tx_builder,
    )

    assert execution.vault == orderly_vault
    assert execution.tx_builder == orderly_tx_builder
    assert execution.broker_id == broker_id
    assert execution.orderly_account_id == orderly_account_id

    # Test vault validation
    execution.check_valid()
    assert execution.vault.contract is not None
    assert execution.vault.address is not None

    # Test routing state details include vault and Orderly information
    details = execution.get_routing_state_details()
    assert "vault" in details
    assert details["vault"] == orderly_vault
    assert "broker_id" in details
    assert details["broker_id"] == broker_id
    assert "orderly_account_id" in details
    assert details["orderly_account_id"] == orderly_account_id

    # Test routing model creation
    routing_model = execution.create_default_routing_model(orderly_strategy_universe)
    assert isinstance(routing_model, OrderlyRouting)
    assert routing_model.vault == orderly_vault
    assert routing_model.broker_id == broker_id
    assert routing_model.orderly_account_id == orderly_account_id

    # Test that the routing model has proper configuration
    reserve_asset = orderly_strategy_universe.get_reserve_asset()
    assert routing_model.reserve_token_address == reserve_asset.address


@pytest.mark.skip(reason="Requires proper Orderly vault deployment on testnet")
def test_orderly_execution_pre_execute_assertions(
    orderly_execution_model: OrderlyExecution,
):
    """Test pre-execution assertions."""
    import datetime
    from tradeexecutor.strategy.routing import RoutingState
    
    ts = datetime.datetime.utcnow()
    routing_state = RoutingState()
    routing_model = None
    
    # This should not raise any assertions
    OrderlyExecution.pre_execute_assertions(
        ts=ts,
        routing_model=routing_model,
        routing_state=routing_state,
    )
    
    # Test with wrong types should raise
    with pytest.raises(AssertionError):
        OrderlyExecution.pre_execute_assertions(
            ts="not a datetime",
            routing_model=routing_model,
            routing_state=routing_state,
        )
    
    with pytest.raises(AssertionError):
        OrderlyExecution.pre_execute_assertions(
            ts=ts,
            routing_model=routing_model,
            routing_state="not a routing state",
        )