"""Check we get Enzyme specific parameters over the webhook API."""
import datetime
from queue import Queue

import pytest
import requests

from eth_defi.enzyme.deployment import POLYGON_DEPLOYMENT
from eth_defi.enzyme.vault import Vault
from tradingstrategy.chain import ChainId

from tradeexecutor.cli.bootstrap import create_metadata
from tradeexecutor.cli.log import setup_in_memory_logging, get_ring_buffer_handler
from tradeexecutor.state.metadata import Metadata
from tradeexecutor.state.state import State
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.strategy.execution_model import AssetManagementMode
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.webhook.server import create_webhook_server


@pytest.fixture()
def store() -> JSONFileStore:
    """Dummy state and store for the tests."""
    portfolio = Portfolio()
    state = State(portfolio=portfolio)
    store = JSONFileStore("/tmp/webhook-test.json")
    store.sync(state)
    return store


@pytest.fixture()
def server_url(store, vault):
    execution_state = RunState()
    execution_state.source_code = "Foobar"
    execution_state.visualisation.small_image = b"1"
    execution_state.visualisation.large_image = b"2"
    execution_state.visualisation.small_image_dark = b"3"
    execution_state.visualisation.large_image_dark = b"4"
    execution_state.version.tag = "v1"
    execution_state.version.commit_message = "Foobar"

    queue = Queue()
    metadata = create_metadata(
        "Foobar",
        "Short desc",
        "Long desc",
        "https://place-puppy.com/300x300",
        asset_management_mode=AssetManagementMode.enzyme,
        chain_id=ChainId.polygon,
        vault=vault)
    server = create_webhook_server("127.0.0.1", 5000, "test", "test", queue, store, metadata, execution_state)
    server_url = "http://test:test@127.0.0.1:5000"
    yield server_url
    server.shutdown()


def test_enzyme_metadata(logger, server_url, vault: Vault):
    """Get executor metadata"""
    resp = requests.get(f"{server_url}/metadata")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Foobar"
    assert data["short_description"] == "Short desc"
    assert data["executor_running"] == True
    assert data["on_chain_data"]["asset_management_mode"] == "enzyme"
    assert data["on_chain_data"]["chain_id"] == 137
    assert data["on_chain_data"]["smart_contracts"]["vault"] == vault.vault.address
    assert data["on_chain_data"]["smart_contracts"]["comptroller"] == vault.comptroller.address
    assert data["on_chain_data"]["smart_contracts"]["generic_adapter"] == vault.generic_adapter.address
    assert data["on_chain_data"]["smart_contracts"]["fund_value_calculator"] == vault.deployment.contracts.fund_value_calculator.address
    # TODO: Not covered in tests yet
    assert data["on_chain_data"]["smart_contracts"]["payment_forwarder"] == None

