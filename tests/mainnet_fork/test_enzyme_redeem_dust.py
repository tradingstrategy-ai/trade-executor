"""Test Enzyme redemption where the redemption request has a closed position with dust."""
import os
import secrets
from pathlib import Path

import pytest
from _pytest.fixtures import FixtureRequest

from eth_defi.provider.anvil import AnvilLaunch, launch_anvil

from tradeexecutor.cli.main import app


pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_POLYGON") or not os.environ.get("TRADING_STRATEGY_API_KEY"), reason="Set JSON_RPC_POLYGON and TRADING_STRATEGY_API_KEY environment variables to run this test")


@pytest.fixture(scope="module")
def end_block() -> int:
    """The chain point of time when we simulate the events."""
    block_time = 2
    days = 6
    return 62514843 - days*24*3600//block_time


@pytest.fixture()
def anvil(request: FixtureRequest, end_block) -> AnvilLaunch:
    mainnet_rpc = os.environ["JSON_RPC_POLYGON"]
    anvil = launch_anvil(
        mainnet_rpc,
        fork_block_number=end_block,  # The timestamp on when the broken position was created
    )
    try:
        yield anvil
    finally:
        #anvil.close(log_level=logging.INFO)
        anvil.close()


@pytest.fixture()
def state_file() -> Path:
    p = Path(os.path.join(os.path.dirname(__file__), "redeem-dust.json"))
    assert p.exists(), f"{p} missing"
    return p


@pytest.fixture()
def strategy_file() -> Path:
    """The strategy module where the broken accounting happened."""
    p = Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "enzyme-polygon-eth-btc-rsi.py"))
    assert p.exists(), f"{p.resolve()} missing"
    return p


@pytest.fixture()
def environment(
    anvil: AnvilLaunch,
    state_file: Path,
    strategy_file: Path,
    end_block: int,
    persistent_test_cache_path,
    ) -> dict:
    """Used by CLI commands, for setting up this test environment"""
    environment = {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": "0x" + secrets.token_bytes(32).hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "enzyme",
        "UNIT_TESTING": "true",
        "UNIT_TEST_FORCE_ANVIL": "true",  # check-wallet command legacy hack
        "LOG_LEVEL": "disabled",
        # "LOG_LEVEL": "info",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "VAULT_ADDRESS": "0xbba6B781e0BAC1798e4E715ef9b1113Bf2387544",
        "VAULT_ADAPTER_ADDRESS": "0x519f26bE61889656e83262ab56D75f00DFDAAEc1",
        "VAULT_PAYMENT_FORWARDER_ADDRESS": "0x638241c16aB5002298B24ED2F0074B4662042258",
        "VAULT_DEPLOYMENT_BLOCK_NUMBER": "60554042",
        "SKIP_SAVE": "true",
        "PROCESS_REDEMPTION": "true",  # Especially test for the broken redemption event
        "PROCESS_REDEMPTION_END_BLOCK_HINT": str(end_block),
        "AUTO_APPROVE": "true",   # Disable interactivity for repair command
        "CACHE_PATH": persistent_test_cache_path,
    }
    return environment


@pytest.mark.slow_test_group
def test_enzyme_redeem_dust(
    environment: dict,
    mocker,
):
    """Test Enzyme redemption where the redemption request has a closed position with dust.

    - The state file contains a closed aPolUSDC position

    - This position has small dust value left

    - Enzyme redemption request tries to redeem part of this dust (1 unit of aPolUSDC)

    - Make sure we can handle the dust redemption request on a closed position

    The problematic SharesRedeemed event:

    .. code-block:: text

        enzyme-polygon-eth-btc-rsi  | tradeexecutor.ethereum.enzyme.vault.UnknownAsset: Asset <aPolUSDC at 0x625e7708f30ca75bfd92586e17077590c60eb4cd> does not map to any open position.

        enzyme-polygon-eth-btc-rsi  | Do not know how to recover. You need to stop trade-executor and run accounting correction.
        enzyme-polygon-eth-btc-rsi  | Could not process redemption event.
        enzyme-polygon-eth-btc-rsi  | Redeemed assets:
        enzyme-polygon-eth-btc-rsi  | <USD Coin (PoS) (USDC) at 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174, 6 decimals, on chain 137>: 84614952
        enzyme-polygon-eth-btc-rsi  | <Wrapped Ether (WETH) at 0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619, 18 decimals, on chain 137>: 0
        enzyme-polygon-eth-btc-rsi  | <(PoS) Wrapped BTC (WBTC) at 0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6, 8 decimals, on chain 137>: 0
        enzyme-polygon-eth-btc-rsi  | <Aave Polygon USDC (aPolUSDC) at 0x625E7708f30cA75bfd92586e17077590C60eb4cD, 6 decimals, on chain 137>: 1
        enzyme-polygon-eth-btc-rsi  | EVM event data:
        enzyme-polygon-eth-btc-rsi  | { 'address': '0xd4d6e8a69a6d4bebbb96188cfc6465d91ae461d6',
        enzyme-polygon-eth-btc-rsi  |   'blockHash': '0xfa29b55146628d84f3b2990a2458e18064e2daa4427112ace270c40fd879f9d8',
        enzyme-polygon-eth-btc-rsi  |   'blockNumber': 62094345,
        enzyme-polygon-eth-btc-rsi  |   'chunk_id': 62093612,
        enzyme-polygon-eth-btc-rsi  |   'context': None,
        enzyme-polygon-eth-btc-rsi  |   'data': '0x000000000000000000000000000000000000000000000004e3f298eb47ba54fb0000000000000000000000000000000000000000000000000000000000000060000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000040000000000000000000000002791bca1f2de4661ed88a30c99a7a9449aa841740000000000000000000000007ceb23fd6bc0add59e62ac25578270cff1b9f6190000000000000000000000001bfd67037b42cf73acf2047067bd4f2c47d9bfd6000000000000000000000000625e7708f30ca75bfd92586e17077590c60eb4cd000000000000000000000000000000000000000000000000000000000000000400000000000000000000000000000000000000000000000000000000050b1f28000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001',
        enzyme-polygon-eth-btc-rsi  |   'event': <class 'web3._utils.datatypes.SharesRedeemed'>,
        enzyme-polygon-eth-btc-rsi  |   'logIndex': '0x76',
        enzyme-polygon-eth-btc-rsi  |   'removed': False,
        enzyme-polygon-eth-btc-rsi  |   'timestamp': 1726917217,
        enzyme-polygon-eth-btc-rsi  |   'topics': [ HexBytes('0xbf88879a1555e4d7d38ebeffabce61fdf5e12ea0468abf855a72ec17b432bed5'),
        enzyme-polygon-eth-btc-rsi  |               HexBytes('0x000000000000000000000000c37b40abdb939635068d3c5f13e7faf686f03b65'),
        enzyme-polygon-eth-btc-rsi  |               HexBytes('0x000000000000000000000000c37b40abdb939635068d3c5f13e7faf686f03b65')],
        enzyme-polygon-eth-btc-rsi  |   'transactionHash': '0x0ac5a9eb7f426d3da655549c539bbad6919a9f6addbae3527d33725859e2e02c',
        enzyme-polygon-eth-btc-rsi  |   'transactionIndex': '0x1f'}

    When running with log level info, the test should spit out warning:

    .. code-block:: text

        2024-10-01 23:03:34 tradeexecutor.ethereum.enzyme.vault                WARNING  Enzyme dust redemption detected and ignored.
        Asset: <aPolUSDC at 0x625e7708f30ca75bfd92586e17077590c60eb4cd>
        Epsilon: 0.1000000000000000055511151231257827021181583404541015625
        Amount: 0.000001


    """

    mocker.patch.dict("os.environ", environment, clear=True)

    app(["repair"], standalone_mode=False)

    with pytest.raises(SystemExit) as sys_exit:
        app(["correct-accounts"], standalone_mode=False)
    assert sys_exit.value.code == 0


