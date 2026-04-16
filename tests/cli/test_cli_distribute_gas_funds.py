"""Test distribute-gas-funds CLI command in dry-run mode.

Uses the CCTP bridge strategy (Arbitrum + Base) with Anvil forks.

To run:

.. code-block:: shell

    source .local-test.env && poetry run pytest tests/cli/test_cli_distribute_gas_funds.py -v --log-cli-level=info

Requires environment variables:
- JSON_RPC_ARBITRUM
- JSON_RPC_BASE
- TRADING_STRATEGY_API_KEY
"""

import os
import secrets
from contextlib import redirect_stdout
from decimal import Decimal
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest
from web3 import Web3, HTTPProvider

from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.cli.main import app
from tradeexecutor.utils.hex import hexbytes_to_hex_str


pytestmark = pytest.mark.skipif(
    not os.environ.get("JSON_RPC_ARBITRUM")
    or not os.environ.get("JSON_RPC_BASE")
    or not os.environ.get("TRADING_STRATEGY_API_KEY"),
    reason="Set JSON_RPC_ARBITRUM, JSON_RPC_BASE and TRADING_STRATEGY_API_KEY to run this test",
)


@pytest.fixture()
def hot_wallet_private_key() -> str:
    return hexbytes_to_hex_str(secrets.token_bytes(32))


@pytest.fixture()
def hot_wallet_address(hot_wallet_private_key) -> str:
    hw = HotWallet.from_private_key(hot_wallet_private_key)
    return hw.address


@pytest.fixture()
def anvil_arbitrum(hot_wallet_address) -> AnvilLaunch:
    """Fork Arbitrum mainnet via Anvil and fund hot wallet with ETH."""
    mainnet_rpc = os.environ["JSON_RPC_ARBITRUM"]
    anvil = launch_anvil(mainnet_rpc)
    try:
        web3 = Web3(HTTPProvider(anvil.json_rpc_url))
        web3.eth.send_transaction({
            "from": web3.eth.accounts[0],
            "to": hot_wallet_address,
            "value": 10 * 10**18,
        })
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def anvil_base(hot_wallet_address) -> AnvilLaunch:
    """Fork Base mainnet via Anvil and fund hot wallet with ETH."""
    mainnet_rpc = os.environ["JSON_RPC_BASE"]
    anvil = launch_anvil(mainnet_rpc)
    try:
        web3 = Web3(HTTPProvider(anvil.json_rpc_url))
        web3.eth.send_transaction({
            "from": web3.eth.accounts[0],
            "to": hot_wallet_address,
            "value": 10 * 10**18,
        })
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def strategy_file() -> Path:
    return Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "cctp_bridge_strategy.py"))


@pytest.fixture()
def environment(
    anvil_arbitrum: AnvilLaunch,
    anvil_base: AnvilLaunch,
    strategy_file: Path,
    hot_wallet_private_key: str,
    tmp_path: Path,
) -> dict:
    return {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet_private_key,
        "JSON_RPC_ARBITRUM": anvil_arbitrum.json_rpc_url,
        "JSON_RPC_BASE": anvil_base.json_rpc_url,
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "DRY_RUN": "true",
        "MIN_GAS_USD": "1000000",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "CACHE_PATH": tmp_path.as_posix(),
    }


def test_distribute_gas_funds_dry_run(environment: dict):
    """distribute-gas-funds dry-run mode displays balances and proposed swaps without external quote dependency.

    The Li.Fi quote API is mocked because dry-run control-flow coverage should
    not fail CI when the external quote endpoint is slow or unavailable.

    1. Patch the process environment for a dry-run CCTP bridge strategy.
    2. Mock the Li.Fi balance and quote boundaries.
    3. Run the command and verify it reaches the dry-run branch.
    """
    quote = mock.MagicMock()
    quote.execution_duration = 30
    quote.get_age_seconds.return_value = 0
    swap = mock.MagicMock()
    swap.source_chain_id = 42161
    swap.target_chain_id = 8453
    swap.target_balance_usd = Decimal("0")
    swap.from_amount_usd = Decimal("10")
    swap.from_amount_raw = 10**16
    swap.quote = quote

    def display_mock_balances(*args, **kwargs):
        print("\nCurrent gas balances:")
        print("mocked balances")

    f = StringIO()
    # 1. Patch the process environment for a dry-run CCTP bridge strategy.
    # 2. Mock the Li.Fi balance and quote boundaries.
    # 3. Run the command and verify it reaches the dry-run branch.
    with mock.patch.dict("os.environ", environment, clear=True):
        with mock.patch(
            "tradeexecutor.cli.commands.distribute_gas_funds.display_balances",
            side_effect=display_mock_balances,
        ):
            with mock.patch(
                "tradeexecutor.cli.commands.distribute_gas_funds.prepare_crosschain_swaps",
                return_value=[swap],
            ) as prepare_swaps:
                with redirect_stdout(f):
                    app(["distribute-gas-funds"], standalone_mode=False)

    output = f.getvalue()
    assert "Current gas balances" in output
    assert "Proposed swaps" in output
    assert "Dry run mode" in output
    prepare_swaps.assert_called_once()
