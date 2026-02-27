"""Test CCTP bridge + satellite chain Uniswap v3 via CLI start command.

Run 5 cycles of the CCTP bridge test strategy on forked Arbitrum + Base:

1. Bridge 5000 USDC from Arbitrum → Base via CCTP
2. Spoof CCTP attestation, swap bridged USDC → WETH on Base via Uniswap v3
3. Sell WETH → USDC on Base
4. Bridge ~4800 USDC from Base → Arbitrum via reverse CCTP
5. Spoof CCTP attestation on Arbitrum, verify funds returned and equity

To run:

.. code-block:: shell

    source .local-test.env && poetry run pytest tests/ethereum/test_cctp_bridge_start.py -v --log-cli-level=info

Requires environment variables:
- JSON_RPC_ARBITRUM
- JSON_RPC_BASE
- TRADING_STRATEGY_API_KEY
"""

import os
import secrets
from decimal import Decimal
from pathlib import Path
from unittest import mock

import pytest
from web3 import Web3, HTTPProvider

from eth_defi.cctp.constants import CCTP_DOMAIN_ARBITRUM, CCTP_DOMAIN_BASE
from eth_defi.cctp.receive import prepare_receive_message
from eth_defi.cctp.testing import craft_cctp_message, forge_attestation, replace_attester_on_fork
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.token import USDC_NATIVE_TOKEN, USDC_WHALE, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.utils.hex import hexbytes_to_hex_str


pytestmark = pytest.mark.skipif(
    not os.environ.get("JSON_RPC_ARBITRUM")
    or not os.environ.get("JSON_RPC_BASE")
    or not os.environ.get("TRADING_STRATEGY_API_KEY"),
    reason="Set JSON_RPC_ARBITRUM, JSON_RPC_BASE and TRADING_STRATEGY_API_KEY to run this test",
)

#: How much USDC to fund the hot wallet with on Arbitrum
FUND_AMOUNT = 10_000 * 10**6  # 10,000 USDC


@pytest.fixture()
def hot_wallet_private_key() -> str:
    """Generate a fresh private key for the hot wallet."""
    return hexbytes_to_hex_str(secrets.token_bytes(32))


@pytest.fixture()
def hot_wallet_address(hot_wallet_private_key) -> str:
    """Derive the hot wallet address from the private key."""
    hw = HotWallet.from_private_key(hot_wallet_private_key)
    return hw.address


@pytest.fixture()
def anvil_arbitrum(hot_wallet_address) -> AnvilLaunch:
    """Fork Arbitrum mainnet via Anvil."""
    mainnet_rpc = os.environ["JSON_RPC_ARBITRUM"]
    arb_usdc_whale = USDC_WHALE[42161]
    anvil = launch_anvil(
        mainnet_rpc,
        unlocked_addresses=[arb_usdc_whale],
    )
    try:
        # Fund hot wallet with ETH and USDC on Arbitrum
        web3 = Web3(HTTPProvider(anvil.json_rpc_url))

        # ETH for gas
        web3.eth.send_transaction({
            "from": web3.eth.accounts[0],
            "to": hot_wallet_address,
            "value": 10 * 10**18,
        })

        # USDC
        usdc_address = USDC_NATIVE_TOKEN[42161]
        usdc = fetch_erc20_details(web3, usdc_address, chain_id=42161)
        tx_hash = usdc.contract.functions.transfer(
            hot_wallet_address, FUND_AMOUNT,
        ).transact({"from": arb_usdc_whale, "gas": 100_000})
        assert_transaction_success_with_explanation(web3, tx_hash)

        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def anvil_base(hot_wallet_address) -> AnvilLaunch:
    """Fork Base mainnet via Anvil."""
    mainnet_rpc = os.environ["JSON_RPC_BASE"]
    anvil = launch_anvil(mainnet_rpc)
    try:
        # Fund hot wallet with ETH on Base (for gas on cycle 3)
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
    p = Path(__file__).resolve().parent / ".." / ".." / "strategies" / "test_only" / "cctp_bridge_start_test.py"
    assert p.exists(), f"Strategy file missing: {p.resolve()}"
    return p


@pytest.fixture()
def environment(
    anvil_arbitrum: AnvilLaunch,
    anvil_base: AnvilLaunch,
    strategy_file: Path,
    hot_wallet_private_key: str,
    tmp_path: Path,
    persistent_test_cache_path,
) -> dict:
    """Environment variables passed to init and start commands."""
    state_file = tmp_path / "test_cctp_bridge_start.json"

    return {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet_private_key,
        "JSON_RPC_ARBITRUM": anvil_arbitrum.json_rpc_url,
        "JSON_RPC_BASE": anvil_base.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "CACHE_PATH": persistent_test_cache_path,
        "CHECK_ACCOUNTS": "false",
        "RUN_SINGLE_CYCLE": "true",
        "MIN_GAS_BALANCE": "0.0",
        "SYNC_TREASURY_ON_STARTUP": "true",
    }


@pytest.mark.slow_test_group
def test_cctp_bridge_start_single_cycle(
    environment: dict,
    anvil_arbitrum: AnvilLaunch,
    anvil_base: AnvilLaunch,
    hot_wallet_address: str,
):
    """Run 5 cycles: bridge → buy → sell → bridge back → verify.

    1. Fork Arbitrum and Base with Anvil
    2. Cycle 1: Strategy bridges 5000 USDC to Base via CCTP depositForBurn
    3. Spoof CCTP attestation on Base fork (mint USDC to hot wallet)
    4. Cycle 2: Swap bridged USDC → WETH on Base via Uniswap v3
    5. Cycle 3: Sell WETH → USDC on Base
    6. Cycle 4: Bridge ~4800 USDC from Base → Arbitrum via reverse CCTP
    7. Spoof CCTP attestation on Arbitrum fork
    8. Cycle 5: No-op, verify funds returned and portfolio equity
    """

    # === Init ===
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["init"], standalone_mode=False)

    state_file = environment["STATE_FILE"]

    # === Cycle 1: Bridge USDC from Arbitrum → Base ===
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["start"], standalone_mode=False)

    state = State.from_json(open(state_file, "rt").read())

    # Should have one open bridge position
    assert len(state.portfolio.open_positions) == 1, \
        f"Expected 1 open position after cycle 1, got {len(state.portfolio.open_positions)}"
    bridge_pos = list(state.portfolio.open_positions.values())[0]
    assert bridge_pos.pair.is_cctp_bridge(), \
        f"Expected CCTP bridge position, got {bridge_pos.pair}"

    # Bridge trade should be successful
    assert len(bridge_pos.trades) == 1
    bridge_trade = list(bridge_pos.trades.values())[0]
    assert bridge_trade.get_status() == TradeStatus.success, \
        f"Bridge trade status: {bridge_trade.get_status()}"

    # Verify USDC was burned on Arbitrum (reserve decreased)
    arb_web3 = Web3(HTTPProvider(anvil_arbitrum.json_rpc_url))
    arb_usdc = fetch_erc20_details(arb_web3, USDC_NATIVE_TOKEN[42161], chain_id=42161)
    arb_balance = arb_usdc.fetch_balance_of(hot_wallet_address)
    assert arb_balance == Decimal(5000), \
        f"Expected 5000 USDC remaining on Arbitrum, got {arb_balance}"

    equity_after_c1 = state.portfolio.get_total_equity()
    # Bridge positions have 0 equity, so total = reserves only (~5000)
    assert equity_after_c1 == pytest.approx(5000, abs=500), \
        f"Equity after cycle 1 should be ~5000 (reserves only), got {equity_after_c1}"

    # === Spoof CCTP attestation on Base ===
    base_web3 = Web3(HTTPProvider(anvil_base.json_rpc_url))

    # Replace the real CCTP attester with our test attester
    base_test_attester = replace_attester_on_fork(base_web3)

    # Craft the CCTP message (simulating what Circle's attestation service returns)
    message = craft_cctp_message(
        source_domain=CCTP_DOMAIN_ARBITRUM,
        destination_domain=CCTP_DOMAIN_BASE,
        nonce=1,
        mint_recipient=hot_wallet_address,
        amount=5000 * 10**6,  # 5000 USDC
        burn_token=USDC_NATIVE_TOKEN[42161],
    )

    # Forge attestation with our test attester
    attestation = forge_attestation(message, base_test_attester)

    # Call receiveMessage on Base to mint USDC
    receive_fn = prepare_receive_message(base_web3, message, attestation)
    tx_hash = receive_fn.transact({"from": base_web3.eth.accounts[0]})
    assert_transaction_success_with_explanation(base_web3, tx_hash)

    # Verify USDC arrived on Base
    base_usdc = fetch_erc20_details(base_web3, USDC_NATIVE_TOKEN[8453], chain_id=8453)
    base_usdc_balance = base_usdc.fetch_balance_of(hot_wallet_address)
    assert base_usdc_balance == Decimal(5000), \
        f"Expected 5000 USDC on Base after CCTP receive, got {base_usdc_balance}"

    # === Cycle 2: Swap bridged USDC → WETH on Base via Uniswap v3 ===
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["start"], standalone_mode=False)

    state = State.from_json(open(state_file, "rt").read())

    # Should have WETH position (and bridge position may still be open)
    weth_positions = [
        pos for pos in state.portfolio.open_positions.values()
        if pos.pair.base.token_symbol == "WETH"
    ]
    assert len(weth_positions) == 1, \
        f"Expected 1 WETH position after cycle 2, got {len(weth_positions)}"

    weth_pos = weth_positions[0]
    assert len(weth_pos.trades) == 1
    weth_trade = list(weth_pos.trades.values())[0]
    assert weth_trade.get_status() == TradeStatus.success, \
        f"WETH trade status: {weth_trade.get_status()}"

    # Verify WETH arrived on Base
    from eth_defi.token import WRAPPED_NATIVE_TOKEN
    base_weth = fetch_erc20_details(base_web3, WRAPPED_NATIVE_TOKEN[8453], chain_id=8453)
    weth_balance = base_weth.fetch_balance_of(hot_wallet_address)
    assert weth_balance > 0, \
        f"Expected WETH balance > 0 on Base, got {weth_balance}"

    # Verify USDC was spent on Base (should be less than 5000 after swap)
    base_usdc_balance_after = base_usdc.fetch_balance_of(hot_wallet_address)
    assert base_usdc_balance_after < Decimal(5000), \
        f"Expected USDC to decrease after swap, got {base_usdc_balance_after}"

    equity_after_c2 = state.portfolio.get_total_equity()
    # Equity = reserves (Arb USDC) + WETH position value; bridge = 0
    assert equity_after_c2 > 4000, \
        f"Equity after cycle 2 should be > 4000 (reserves + WETH), got {equity_after_c2}"

    # === Cycle 3: Sell WETH → USDC on Base ===
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["start"], standalone_mode=False)

    state = State.from_json(open(state_file, "rt").read())

    # WETH position should be closed
    weth_closed = [
        pos for pos in state.portfolio.closed_positions.values()
        if pos.pair.base.token_symbol == "WETH"
    ]
    assert len(weth_closed) == 1, \
        f"Expected 1 closed WETH position after cycle 3, got {len(weth_closed)}"

    # WETH sell trade should be successful
    weth_sell_trade = list(weth_closed[0].trades.values())[-1]
    assert weth_sell_trade.get_status() == TradeStatus.success, \
        f"WETH sell trade status: {weth_sell_trade.get_status()}"

    # Verify WETH was sold (balance should be ~0)
    weth_balance_after_sell = base_weth.fetch_balance_of(hot_wallet_address)
    assert weth_balance_after_sell < weth_balance, \
        f"Expected WETH balance to decrease after sell, got {weth_balance_after_sell}"

    # Verify USDC_Base recovered on Base
    base_usdc_after_sell = base_usdc.fetch_balance_of(hot_wallet_address)
    assert base_usdc_after_sell > base_usdc_balance_after, \
        f"Expected USDC to increase after WETH sell, got {base_usdc_after_sell}"

    equity_after_c3 = state.portfolio.get_total_equity()
    assert equity_after_c3 > 4000, \
        f"Equity after cycle 3 should be > 4000, got {equity_after_c3}"

    # === Cycle 4: Bridge USDC from Base → Arbitrum ===
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["start"], standalone_mode=False)

    state = State.from_json(open(state_file, "rt").read())

    # Should have a reverse bridge position (source chain = Base)
    reverse_bridge_positions = [
        pos for pos in state.portfolio.open_positions.values()
        if pos.pair.is_cctp_bridge() and pos.pair.quote.chain_id == 8453
    ]
    assert len(reverse_bridge_positions) == 1, \
        f"Expected 1 reverse bridge position after cycle 4, got {len(reverse_bridge_positions)}"

    reverse_bridge_trade = list(reverse_bridge_positions[0].trades.values())[0]
    assert reverse_bridge_trade.get_status() == TradeStatus.success, \
        f"Reverse bridge trade status: {reverse_bridge_trade.get_status()}"

    # Verify USDC was burned on Base
    base_usdc_after_bridge_back = base_usdc.fetch_balance_of(hot_wallet_address)
    assert base_usdc_after_bridge_back < base_usdc_after_sell, \
        f"Expected USDC to decrease on Base after reverse bridge, got {base_usdc_after_bridge_back}"

    # === Spoof CCTP attestation on Arbitrum ===
    arb_test_attester = replace_attester_on_fork(arb_web3)

    # Get the bridged amount from the trade
    bridge_back_amount_raw = int(reverse_bridge_trade.planned_reserve * 10**6)

    arb_message = craft_cctp_message(
        source_domain=CCTP_DOMAIN_BASE,
        destination_domain=CCTP_DOMAIN_ARBITRUM,
        nonce=1,
        mint_recipient=hot_wallet_address,
        amount=bridge_back_amount_raw,
        burn_token=USDC_NATIVE_TOKEN[8453],
    )
    arb_attestation = forge_attestation(arb_message, arb_test_attester)

    arb_receive_fn = prepare_receive_message(arb_web3, arb_message, arb_attestation)
    tx_hash = arb_receive_fn.transact({"from": arb_web3.eth.accounts[0]})
    assert_transaction_success_with_explanation(arb_web3, tx_hash)

    # Verify USDC arrived on Arbitrum
    arb_usdc_after_return = arb_usdc.fetch_balance_of(hot_wallet_address)
    assert arb_usdc_after_return > Decimal(5000), \
        f"Expected > 5000 USDC on Arb after CCTP receive back, got {arb_usdc_after_return}"

    # === Cycle 5: No-op — verify final state ===
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["start"], standalone_mode=False)

    state = State.from_json(open(state_file, "rt").read())

    # Treasury sync should have detected the returned USDC on Arb
    final_equity = state.portfolio.get_total_equity()
    # Expect ~10000 minus swap fees (~0.05% * 2 swaps * ~4900)
    # Use generous tolerance for slippage + market price changes during fork
    assert final_equity > 8000, \
        f"Final equity {final_equity} should be > 8000 (started with 10000)"

    # Verify on-chain Arbitrum balance increased from the bridge back
    final_arb_balance = arb_usdc.fetch_balance_of(hot_wallet_address)
    assert final_arb_balance > Decimal(9000), \
        f"Expected > 9000 USDC on Arb after round trip, got {final_arb_balance}"
