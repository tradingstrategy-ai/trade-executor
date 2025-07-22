"""Test trade execution store persistence
"""
import pytest
import logging
import brotli
from pathlib import Path

from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.store import JSONFileStore


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request)


@pytest.fixture
def state_and_store(tmp_path, mocker):
    """Create a mock state and JSONFileStore for testing"""
    test_file = tmp_path / "test.json"
    store = JSONFileStore(test_file)

    mock_state = mocker.Mock()
    mock_state.to_dict.return_value = {"test": "data", "nested": {"value": 123}}

    return mock_state, store


def test_sync_creates_both_files(caplog, state_and_store):
    """Test that sync creates both .json and .json.br files"""
    mock_state, store = state_and_store

    with caplog.at_level(logging.INFO):
        store.sync(mock_state)

    # Check files exist
    assert store.path.exists()
    br_path = Path(f"{store.path}.br")
    assert br_path.exists()

    # Check logging
    assert "Saved state to" in caplog.text
    assert "Saved compressed state to" in caplog.text
    assert len(caplog.records) == 2


def test_compressed_file_contains_valid_data(state_and_store):
    """Test that .json.br file can be decompressed to original JSON"""
    mock_state, store = state_and_store

    store.sync(mock_state)

    # Read original JSON
    original_json = store.path.read_text()

    # Read and decompress .br file
    br_path = Path(f"{store.path}.br")
    with open(br_path, 'rb') as f:
        compressed_data = f.read()

    decompressed = brotli.decompress(compressed_data).decode('utf-8')

    # Should be identical
    assert decompressed == original_json


def test_compressed_file_timestamp(state_and_store):
    """Test that .json.br file timestamp is >= .json file timestamp"""
    mock_state, store = state_and_store

    store.sync(mock_state)

    json_mtime = store.path.stat().st_mtime
    br_mtime = Path(f"{store.path}.br").stat().st_mtime

    # .json.br file should be same age or newer (for Caddy compatibility)
    assert br_mtime >= json_mtime
