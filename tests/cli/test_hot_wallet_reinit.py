import os
from pathlib import Path
from unittest.mock import patch

import pytest
from eth_typing import HexAddress

from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from tradeexecutor import cli
from tradeexecutor.state.state import State
from tradingstrategy.pair import PandasPairUniverse


pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test module")


@pytest.fixture()
def environment(
    anvil: AnvilLaunch,
    deployer: HexAddress,
    user_1: HexAddress,
    uniswap_v2: UniswapV2Deployment,
    pair_universe: PandasPairUniverse,
    hot_wallet: HotWallet,
    state_file: Path,
    strategy_file: Path,
    ) -> dict:
    """Set up environment vars for all CLI commands."""

    environment = {
        "EXECUTOR_ID": "test_close_all",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "TEST_EVM_UNISWAP_V2_ROUTER": uniswap_v2.router.address,
        "TEST_EVM_UNISWAP_V2_FACTORY": uniswap_v2.factory.address,
        "TEST_EVM_UNISWAP_V2_INIT_CODE_HASH": uniswap_v2.init_code_hash,
        "CONFIRMATION_BLOCK_COUNT": "0",  # Needed for test backend, Anvil
    }
    return environment

@pytest.fixture()
def backup_file() -> str:
    return "/tmp/test_hot_wallet_live_trading_reinit.reinit-backup-1.json"


def test_hot_wallet_live_trading_reinit(
        multipair_environment: dict,
        state_file: Path,
        vault,
        deployer,
        usdc,
        backup_file,
):
    """Reset a strategy state

    - Any open positions are correctly recovered

    - The strategy history is cleared
    """

    if os.path.exists(backup_file):
        os.remove(backup_file)

    # First init the strategy
    with patch.dict(os.environ, multipair_environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["init"])
        assert e.value.code == 0

    # Check we have balance in our hot wallet

    # Make a buy

    # Reset
    with patch.dict(os.environ, multipair_environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["reset"])
        assert e.value.code == 0

    assert os.path.exists(backup_file)

    # See that the reinitialised state looks correct
    with state_file.open("rt") as inp:
        state: State = State.from_json(inp.read())
        reserve_position = state.portfolio.get_default_reserve_position()
        assert reserve_position.quantity == 500

        treasury = state.sync.treasury
        deployment = state.sync.deployment
        assert deployment.initialised_at
        assert treasury.last_block_scanned > 1
        assert treasury.last_updated_at
        assert len(treasury.balance_update_refs) == 1
        assert len(reserve_position.balance_updates) == 1
