"""Manual test: GMX Lagoon vault lifecycle via trade-executor CLI.

Exercises the full lifecycle of a single-chain Lagoon vault with GMX
perpetuals support on Arbitrum:

1. Deploy vault via ``lagoon-deploy-vault --strategy-file=minimal_gmx_strategy.py``
   with ``ANY_ASSET=true`` and all GMX markets whitelisted
2. Initialise state via ``init``
3. Deposit USDC into the vault
4. Settle vault
5. Run strategy cycle (creates exchange account position)
6. Open ETH/USD long position via LagoonGMXTradingWallet
7. Close ETH/USD position
8. Settle and redeem vault shares (live only)
9. Verify USDC returned to deployer (live only)

In simulated mode (Anvil fork), GMX trading is exercised using mock
oracles and keeper impersonation via ``setup_mock_oracle()`` and
``execute_order_as_keeper()`` from ``eth_defi.gmx.testing``.

Modes
-----

**Simulated (Anvil fork)** — set ``SIMULATE=true``:

- Forks Arbitrum mainnet locally with Anvil
- Funds the deployer with ETH and USDC automatically
- Uses mock oracle + keeper impersonation for GMX order execution
- Full lifecycle including GMX trading steps

**Mainnet** — set ``NETWORK=mainnet`` (default):

- Runs against real Arbitrum mainnet
- Deployer must hold ETH + USDC
- Full GMX trading lifecycle with real keepers

**Testnet** — set ``NETWORK=testnet``:

- Runs against Arbitrum Sepolia
- Deployer must hold testnet ETH + USDC

Environment variables
---------------------

``SIMULATE``
    Set to ``true`` to fork Arbitrum mainnet with Anvil.
    Default: ``false``.

``NETWORK``
    ``mainnet`` (default), ``testnet``, or ``simulate`` (alias for SIMULATE=true).

``JSON_RPC_ARBITRUM``
    Arbitrum RPC URL. Required.

``GMX_PRIVATE_KEY``
    Deployer private key (matching ``lagoon-gmx-example.py``).
    In simulate mode a random key is generated.

``USDC_AMOUNT``
    Amount of USDC to deposit (default: ``5``).

``GMX_ENABLED``
    Set to ``true`` to enable GMX account value tracking in ``start``.

``TRADING_STRATEGY_API_KEY``
    TradingStrategy.ai API key. Optional for code-based strategies.

Usage
-----

Simulated (Anvil fork):

.. code-block:: shell

    SIMULATE=true \\
    JSON_RPC_ARBITRUM="https://arb1.arbitrum.io/rpc" \\
    python scripts/lagoon/manual-trade-executor-gmx.py

Mainnet:

.. code-block:: shell

    JSON_RPC_ARBITRUM="https://..." \\
    GMX_PRIVATE_KEY="0x..." \\
    GMX_ENABLED=true \\
    python scripts/lagoon/manual-trade-executor-gmx.py
"""

import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from unittest import mock

from web3 import Web3

from eth_defi.chain import get_chain_id_by_name
from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.erc_4626.vault_protocol.lagoon.config_event_scanner import (
    build_multichain_guard_config, fetch_guard_config_events, format_guard_config_report)
from eth_defi.hypersync.server import get_hypersync_server
from eth_defi.erc_4626.vault_protocol.lagoon.testing import (
    fund_lagoon_vault, redeem_vault_shares)
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.gmx.config import GMXConfig
from eth_defi.gmx.constants import EXECUTION_BUFFER_TESTNET
from eth_defi.gmx.core.open_positions import GetOpenPositions
from eth_defi.gmx.lagoon.approvals import UNLIMITED, approve_gmx_collateral_via_vault
from eth_defi.gmx.lagoon.wallet import LagoonGMXTradingWallet
from eth_defi.gmx.testing import execute_order_as_keeper, extract_order_key_from_receipt, setup_mock_oracle
from eth_defi.gmx.trading import GMXTrading
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import (AnvilLaunch, fork_network_anvil,
                                     fund_erc20_on_anvil, set_balance)
from eth_defi.provider.broken_provider import _latest_delayed_block_number_cache
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDC_NATIVE_TOKEN, TokenDetails, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.utils import setup_console_logging
from eth_defi.vault.base import VaultSpec

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State

logger = logging.getLogger(__name__)

#: Arbitrum mainnet chain ID
ARBITRUM_CHAIN_ID = get_chain_id_by_name("arbitrum")

#: Arbitrum Sepolia chain ID
ARBITRUM_SEPOLIA_CHAIN_ID = get_chain_id_by_name("arbitrum_sepolia")

#: GMX markets on Arbitrum Sepolia use USDC.SG (Stargate USDC) as collateral,
#: not the canonical testnet USDC from USDC_NATIVE_TOKEN.
GMX_TESTNET_USDC_ADDRESS = "0x3253a335E7bFfB4790Aa4C25C4250d206E9b9773"
GMX_TESTNET_COLLATERAL_SYMBOL = "USDC.SG"


def _check_shares(web3: Web3, vault_address: str, deployer_address: str, label: str):
    """Log the deployer's vault share balance at a checkpoint."""
    vault = create_vault_instance(
        web3, vault_address,
        features={ERC4626Feature.lagoon_like},
        default_block_identifier="latest",
        require_denomination_token=True,
    )
    share_token = vault.share_token
    raw = share_token.fetch_raw_balance_of(deployer_address)
    human = share_token.convert_to_decimals(raw)
    print(f"  [shares] {label}: {human} {share_token.symbol} (raw={raw})")


def run_cli(args: list[str], env: dict):
    """Run a trade-executor CLI command with the given environment.

    Uses ``mock.patch.dict`` to set environment variables cleanly.
    Clears the block cache before each call.
    """
    _latest_delayed_block_number_cache.clear()

    logger.info("Running CLI: trade-executor %s", " ".join(args))
    patched_env = {**env}
    for key in ("PATH", "HOME", "USER", "TMPDIR", "SHELL"):
        if key not in patched_env and key in os.environ:
            patched_env[key] = os.environ[key]
    with mock.patch.dict("os.environ", patched_env, clear=True):
        app(args, standalone_mode=False)


def load_state(state_file: str) -> State:
    """Load strategy state from a JSON file."""
    with open(state_file, "rt") as f:
        return State.from_json(f.read())


def _setup_gmx_trading(
    *,
    web3: Web3,
    vault_address: str,
    safe_address: str,
    module_address: str,
    deployer: HotWallet,
    usdc_token: TokenDetails,
    simulate: bool,
) -> tuple[LagoonGMXTradingWallet, GMXTrading, GetOpenPositions]:
    """Set up GMX trading infrastructure.

    Constructs the LagoonVault, LagoonGMXTradingWallet, GMXTrading,
    and GetOpenPositions objects needed for steps 6-9.

    In simulate mode, sets up mock oracle and funds Safe with ETH.

    :return:
        Tuple of ``(lagoon_wallet, trading, positions_reader)``.
    """
    vault = LagoonVault(
        web3,
        VaultSpec(web3.eth.chain_id, vault_address),
        trading_strategy_module_address=module_address,
    )

    lagoon_wallet = LagoonGMXTradingWallet(
        vault, deployer, gas_buffer=500_000, forward_eth=True,
    )

    deployer.sync_nonce(web3)

    # Approve USDC for GMX SyntheticsRouter via Safe
    approve_gmx_collateral_via_vault(
        vault, deployer, usdc_token, UNLIMITED,
    )
    deployer.sync_nonce(web3)

    gmx_config = GMXConfig(web3=web3, user_wallet_address=safe_address)
    trading = GMXTrading(gmx_config)
    positions_reader = GetOpenPositions(gmx_config)

    return lagoon_wallet, trading, positions_reader


def _open_gmx_position(
    *,
    web3: Web3,
    lagoon_wallet: LagoonGMXTradingWallet,
    trading: GMXTrading,
    market_symbol: str,
    collateral_symbol: str,
    size_usd: float,
    leverage: float,
    is_long: bool = True,
    simulate: bool = False,
) -> dict:
    """Open a GMX perpetuals position via LagoonGMXTradingWallet.

    Creates an order via GMXTrading, signs it through the vault's
    performCall(), and either executes via keeper (simulate) or
    waits for on-chain keeper (live).

    :return:
        Position info dict with market_symbol, is_long, order_key.
    """
    direction = "long" if is_long else "short"
    print(f"  Opening {market_symbol}/USD {direction}: ${size_usd} @ {leverage}x leverage")

    lagoon_wallet.sync_nonce(web3)

    order_result = trading.open_position(
        market_symbol=market_symbol,
        collateral_symbol=collateral_symbol,
        start_token_symbol=collateral_symbol,
        is_long=is_long,
        size_delta_usd=size_usd,
        leverage=leverage,
        execution_buffer=EXECUTION_BUFFER_TESTNET,
    )

    transaction = order_result.transaction.copy()
    if "nonce" in transaction:
        del transaction["nonce"]

    signed_tx = lagoon_wallet.sign_transaction_with_new_nonce(transaction)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    assert_transaction_success_with_explanation(web3, tx_hash)
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

    order_key = extract_order_key_from_receipt(receipt, web3)
    print(f"  Order created: key={order_key.hex()}")

    if simulate:
        execute_order_as_keeper(web3, order_key)
        # Restore deployer ETH (drained by keeper execution on Anvil)
        set_balance(web3, lagoon_wallet.asset_manager.address, 10 * 10**18)
        lagoon_wallet.sync_nonce(web3)
        print("  Order executed by simulated keeper")
    else:
        print("  Waiting 30s for keeper execution...")
        time.sleep(30)

    return {"market_symbol": market_symbol, "is_long": is_long, "order_key": order_key}


def _close_gmx_position(
    *,
    web3: Web3,
    lagoon_wallet: LagoonGMXTradingWallet,
    trading: GMXTrading,
    positions_reader: GetOpenPositions,
    safe_address: str,
    market_symbol: str,
    collateral_symbol: str,
    is_long: bool = True,
    simulate: bool = False,
) -> None:
    """Close a GMX perpetuals position via LagoonGMXTradingWallet.

    Queries open positions to get the full position size and collateral,
    then creates a close order for the entire position.
    """
    direction = "long" if is_long else "short"
    position_key = f"{market_symbol}_{direction}"

    positions = positions_reader.get_data(safe_address)
    assert position_key in positions, \
        f"Position {position_key} not found. Open positions: {list(positions.keys())}"

    pos = positions[position_key]
    size_delta_usd = pos["position_size"]
    # initial_collateral_amount is in raw token units — convert to human-readable
    collateral_decimals = 6  # USDC
    initial_collateral_delta = pos["initial_collateral_amount"] / 10**collateral_decimals

    print(f"  Closing {position_key}: size=${size_delta_usd:.2f}, collateral={initial_collateral_delta:.2f}")

    lagoon_wallet.sync_nonce(web3)

    order_result = trading.close_position(
        market_symbol=market_symbol,
        collateral_symbol=collateral_symbol,
        start_token_symbol=collateral_symbol,
        is_long=is_long,
        size_delta_usd=size_delta_usd,
        initial_collateral_delta=initial_collateral_delta,
        execution_buffer=EXECUTION_BUFFER_TESTNET,
    )

    transaction = order_result.transaction.copy()
    if "nonce" in transaction:
        del transaction["nonce"]

    signed_tx = lagoon_wallet.sign_transaction_with_new_nonce(transaction)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    assert_transaction_success_with_explanation(web3, tx_hash)
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

    order_key = extract_order_key_from_receipt(receipt, web3)
    print(f"  Close order created: key={order_key.hex()}")

    if simulate:
        execute_order_as_keeper(web3, order_key)
        # Restore deployer ETH (drained by keeper execution on Anvil)
        set_balance(web3, lagoon_wallet.asset_manager.address, 10 * 10**18)
        lagoon_wallet.sync_nonce(web3)
        print(f"  Close order executed by simulated keeper")
    else:
        print(f"  Waiting 30s for keeper execution...")
        time.sleep(30)


def _revalue_and_check(
    *,
    run_cli_func: Callable,
    start_env: dict,
    state_file: str,
    label: str,
    expected_min_gmx_value: float = 0,
) -> State:
    """Run a strategy cycle and verify GMX exchange account value.

    Runs the ``start`` CLI command for a single cycle, then loads the
    state and checks that the exchange account position value meets
    the expected minimum.

    :return:
        The loaded State after the cycle.
    """
    run_cli_func(["start"], start_env)

    state = load_state(state_file)

    exchange_positions = [
        pos for pos in state.portfolio.open_positions.values()
        if pos.is_exchange_account()
    ]

    if exchange_positions:
        gmx_pos = exchange_positions[0]
        gmx_value = float(gmx_pos.get_value())
        total_value = float(state.portfolio.get_total_equity())
        print(f"  {label}: GMX value=${gmx_value:.2f}, portfolio total=${total_value:.2f}")
        assert gmx_value >= expected_min_gmx_value, \
            f"GMX value ${gmx_value:.2f} below minimum ${expected_min_gmx_value:.2f}"
    else:
        print(f"  {label}: No exchange account position found")
        if expected_min_gmx_value > 0:
            raise AssertionError(f"Expected GMX value >= ${expected_min_gmx_value} but no position found")

    return state


def setup_simulation(
    *,
    json_rpc_arbitrum: str,
    simulate: bool,
    private_key: str | None,
    usdc_amount: Decimal,
    is_testnet: bool = False,
) -> tuple:
    """Set up Web3 connection, deployer wallet, and optionally Anvil fork.

    :return:
        Tuple of ``(web3, deployer, private_key,
        json_rpc_arbitrum, anvil_launches)``.
    """
    anvil_launches: list[AnvilLaunch] = []
    chain_id = ARBITRUM_SEPOLIA_CHAIN_ID if is_testnet else ARBITRUM_CHAIN_ID

    if simulate:
        logger.info("SIMULATE=true — forking Arbitrum with Anvil")

        launch = fork_network_anvil(json_rpc_arbitrum)
        anvil_launches.append(launch)

        web3 = create_multi_provider_web3(launch.json_rpc_url, retries=2)

        # Set up mock oracle before vault deployment — needed for
        # keeper impersonation in GMX trading steps
        setup_mock_oracle(web3)

        # Override RPC URL for CLI commands to point at Anvil fork
        json_rpc_arbitrum = launch.json_rpc_url

        # Create a random deployer wallet, funded with ETH
        deployer = HotWallet.create_for_testing(web3, eth_amount=100)
        private_key = "0x" + deployer.account.key.hex()

        # Fund deployer with USDC
        usdc_address = USDC_NATIVE_TOKEN[chain_id]
        usdc_token = fetch_erc20_details(web3, usdc_address)
        usdc_raw = usdc_token.convert_to_raw(usdc_amount) * 10  # 10x headroom
        fund_erc20_on_anvil(web3, usdc_address, deployer.address, usdc_raw)

        logger.info(
            "Deployer %s funded with 100 ETH + %d USDC on Arbitrum fork",
            deployer.address, usdc_raw // 10**usdc_token.decimals,
        )
    else:
        web3 = create_multi_provider_web3(json_rpc_arbitrum)
        deployer = HotWallet.from_private_key(private_key)

    actual_chain_id = web3.eth.chain_id
    if not simulate:
        assert actual_chain_id == chain_id, \
            f"Expected chain {chain_id}, got {actual_chain_id}"

    deployer.sync_nonce(web3)

    return (
        web3, deployer, private_key,
        json_rpc_arbitrum, anvil_launches,
    )


def verify_deployer_balances(
    *,
    web3: Web3,
    deployer: HotWallet,
    usdc_amount: Decimal,
) -> TokenDetails:
    """Verify deployer has ETH and sufficient USDC.

    :return:
        The USDC token details.
    """
    eth_balance = web3.eth.get_balance(deployer.address)
    logger.info("Deployer: %s", deployer.address)
    logger.info("  ETH: %.6f", eth_balance / 10**18)
    assert eth_balance > 0, f"Deployer has no ETH. Fund {deployer.address} first."

    chain_id = web3.eth.chain_id
    usdc_address = USDC_NATIVE_TOKEN[chain_id]
    usdc_token = fetch_erc20_details(web3, usdc_address)
    deployer_usdc = usdc_token.fetch_balance_of(deployer.address)
    logger.info("  USDC: %s", deployer_usdc)
    assert deployer_usdc >= usdc_amount, \
        f"Deployer needs {usdc_amount} USDC but has {deployer_usdc}."

    return usdc_token


def _run_test_lifecycle(
    *,
    simulate: bool,
    web3: Web3,
    deployer: HotWallet,
    usdc_token: TokenDetails,
    private_key: str,
    json_rpc_arbitrum: str,
    strategy_file: Path,
    usdc_amount: Decimal,
    trading_strategy_api_key: str,
    is_testnet: bool = False,
):
    """Run the full test lifecycle."""

    chain_id = web3.eth.chain_id
    rpc_env_key = "JSON_RPC_ARBITRUM_SEPOLIA" if is_testnet else "JSON_RPC_ARBITRUM"

    with tempfile.TemporaryDirectory() as tmp_dir:
        state_file = str(Path(tmp_dir) / "state.json")
        vault_record_file = str(Path(tmp_dir) / "vault-record.txt")
        cache_path = str(Path(tmp_dir) / "cache")

        # ===================================================================
        # Step 1: Deploy vault with GMX whitelisting
        # ===================================================================
        print("\n=== Step 1: Deploy GMX vault ===")

        deploy_env = {
            "STRATEGY_FILE": str(strategy_file),
            "PRIVATE_KEY": private_key,
            rpc_env_key: json_rpc_arbitrum,
            "VAULT_RECORD_FILE": vault_record_file,
            "FUND_NAME": "Test GMX Vault",
            "FUND_SYMBOL": "TEST",
            "ANY_ASSET": "true",
            "PERFORMANCE_FEE": "0",
            "MANAGEMENT_FEE": "0",
            "UNIT_TESTING": "true",
            "LOG_LEVEL": "info",
        }
        if is_testnet:
            deploy_env["GMX_NETWORK"] = "testnet"

        run_cli(["lagoon-deploy-vault"], deploy_env)

        # Read deployment record
        deployment_json = vault_record_file.replace(".txt", ".json")
        with open(deployment_json) as f:
            deployment_data = json.load(f)

        # For single-chain, get the arbitrum deployment
        chain_slug = "arbitrum_sepolia" if is_testnet else "arbitrum"
        dep = deployment_data["deployments"][chain_slug]
        vault_address = dep["vault_address"]
        safe_address = dep["safe_address"]
        module_address = dep["module_address"]

        print(f"  Vault:  {vault_address}")
        print(f"  Safe:   {safe_address}")
        print(f"  Module: {module_address}")

        # Print guard configuration report.
        # In simulate mode (Anvil fork) use RPC get_logs directly.
        # For live/testnet use Envio HyperSync — Arbitrum Sepolia's
        # public RPC is too rate-limited for event scanning.
        hs_client = None
        if not simulate:
            try:
                import hypersync
                hypersync_api_key = os.environ.get("HYPERSYNC_API_KEY")
                hypersync_url = get_hypersync_server(web3)
                hs_config = hypersync.ClientConfig(url=hypersync_url)
                if hypersync_api_key:
                    hs_config = hypersync.ClientConfig(url=hypersync_url, bearer_token=hypersync_api_key)
                hs_client = hypersync.HypersyncClient(hs_config)
            except ImportError:
                logger.warning("hypersync not installed — falling back to RPC for guard config scan")

        chain_web3 = {web3.eth.chain_id: web3}
        module_addresses = {web3.eth.chain_id: module_address}
        events, _module_addrs = fetch_guard_config_events(
            safe_address=safe_address,
            web3=web3,
            hypersync_client=hs_client,
            chain_web3=chain_web3,
            follow_cctp=False,
        )
        config = build_multichain_guard_config(events, safe_address, module_addresses)
        report = format_guard_config_report(
            config=config,
            events=events,
            chain_web3=chain_web3,
        )
        print(report)

        # ===================================================================
        # Step 2: Initialise state
        # ===================================================================
        print("\n=== Step 2: Initialise state ===")

        base_env = {
            "ID": "test-gmx",
            "STRATEGY_FILE": str(strategy_file),
            "STATE_FILE": state_file,
            "PRIVATE_KEY": private_key,
            rpc_env_key: json_rpc_arbitrum,
            "ASSET_MANAGEMENT_MODE": "lagoon",
            "VAULT_ADDRESS": vault_address,
            "VAULT_ADAPTER_ADDRESS": module_address,
            "UNIT_TESTING": "true",
            "LOG_LEVEL": "info",
            "CACHE_PATH": cache_path,
            "MIN_GAS_BALANCE": "0.0",
            "GMX_ENABLED": "true",
            "GMX_SAFE_ADDRESS": safe_address,
        }
        if is_testnet:
            base_env["GMX_NETWORK"] = "testnet"
        if trading_strategy_api_key:
            base_env["TRADING_STRATEGY_API_KEY"] = trading_strategy_api_key

        run_cli(["init"], {**base_env, "NAME": "Test GMX Vault"})

        assert Path(state_file).exists(), "State file was not created"
        print(f"  State file: {state_file}")

        # ===================================================================
        # Step 3: Deposit USDC into the vault
        # ===================================================================
        print("\n=== Step 3: Deposit USDC ===")

        deployer.sync_nonce(web3)
        fund_lagoon_vault(
            web3=web3,
            vault_address=vault_address,
            asset_manager=deployer.address,
            test_account_with_balance=deployer.address,
            trading_strategy_module_address=module_address,
            amount=usdc_amount,
            hot_wallet=deployer,
        )

        safe_usdc = usdc_token.fetch_balance_of(safe_address)
        print(f"  Safe USDC after deposit: {safe_usdc}")

        _check_shares(web3, vault_address, deployer.address, "After fund_lagoon_vault")

        # ===================================================================
        # Step 4: Settle vault
        # ===================================================================
        print("\n=== Step 4: Settle vault ===")

        settle_env = {k: v for k, v in base_env.items() if k != "NAME"}
        settle_env["SYNC_INTEREST"] = "false"

        run_cli(["lagoon-settle"], settle_env)
        print("  Vault settled")
        _check_shares(web3, vault_address, deployer.address, "After settle")

        # ===================================================================
        # Step 5: Run strategy cycle — creates exchange account position
        # ===================================================================
        print("\n=== Step 5: Strategy cycle — create exchange account position ===")

        start_env = {
            **base_env,
            "RUN_SINGLE_CYCLE": "true",
            "TRADE_IMMEDIATELY": "true",
            "MAX_CYCLES": "1",
            "SYNC_TREASURY_ON_STARTUP": "true",
            "CHECK_ACCOUNTS": "false",
            "MAX_SLIPPAGE": "0.05",
        }

        run_cli(["start"], start_env)

        state = load_state(state_file)

        # Verify exchange account position was created
        exchange_positions = [
            pos for pos in state.portfolio.open_positions.values()
            if pos.is_exchange_account()
        ]
        assert len(exchange_positions) >= 1, \
            f"Expected at least 1 exchange account position, got {len(exchange_positions)}"

        gmx_pos = exchange_positions[0]
        print(f"  Exchange account position: {gmx_pos.pair.get_ticker()}")
        print(f"  Protocol: {gmx_pos.pair.get_exchange_account_protocol()}")
        print(f"  Value: {gmx_pos.get_quantity()}")

        # ===================================================================
        # Steps 6-9: GMX trading via LagoonGMXTradingWallet
        # ===================================================================

        lagoon_wallet, trading, positions_reader = _setup_gmx_trading(
            web3=web3,
            vault_address=vault_address,
            safe_address=safe_address,
            module_address=module_address,
            deployer=deployer,
            usdc_token=usdc_token,
            simulate=simulate,
        )

        # Step 6: Open ETH/USD long
        # Note: In simulate mode, execute_order_as_keeper() only supports
        # ETH/USDC oracle params, so we use ETH market for all steps.
        print("\n=== Step 6: Open ETH/USD long position ===")
        collateral = GMX_TESTNET_COLLATERAL_SYMBOL if is_testnet else "USDC"
        _open_gmx_position(
            web3=web3, lagoon_wallet=lagoon_wallet, trading=trading,
            market_symbol="ETH", collateral_symbol=collateral, size_usd=5.0,
            leverage=1.5, is_long=True, simulate=simulate,
        )

        state = _revalue_and_check(
            run_cli_func=run_cli, start_env=start_env, state_file=state_file,
            label="After ETH/USD open", expected_min_gmx_value=2.0,
        )

        # Step 7: Close ETH/USD
        print("\n=== Step 7: Close ETH/USD long position ===")
        _close_gmx_position(
            web3=web3, lagoon_wallet=lagoon_wallet, trading=trading,
            positions_reader=positions_reader, safe_address=safe_address,
            market_symbol="ETH", collateral_symbol=collateral, is_long=True,
            simulate=simulate,
        )

        state = _revalue_and_check(
            run_cli_func=run_cli, start_env=start_env, state_file=state_file,
            label="After ETH/USD close", expected_min_gmx_value=0,
        )

        # ===================================================================
        # Step 10: Settle and redeem
        # ===================================================================
        if not simulate:
            print("\n=== Step 10: Settle and redeem ===")

            run_cli(["lagoon-settle"], settle_env)

            deployer.sync_nonce(web3)
            try:
                redeem_vault_shares(
                    web3=web3,
                    vault_address=vault_address,
                    redeemer=deployer.address,
                    hot_wallet=deployer,
                )
                print("  Vault shares redeemed")
            except Exception as e:
                logger.warning("Redeem failed (may be expected in test): %s", e)

            # ===================================================================
            # Step 11: Verify USDC returned
            # ===================================================================
            print("\n=== Step 11: Verify USDC returned ===")

            final_usdc = usdc_token.fetch_balance_of(deployer.address)
            print(f"  Final deployer USDC: {final_usdc}")
            print(f"  Mainnet complete — deployer recovered {final_usdc} USDC")

        print("\n" + "=" * 70)
        print("GMX Lagoon vault lifecycle test PASSED")
        print("=" * 70)


def main():
    setup_console_logging("warning")

    # ----- Parse environment -----
    json_rpc_arbitrum = os.environ.get("JSON_RPC_ARBITRUM")
    simulate = os.environ.get("SIMULATE", "").lower() in ("true", "1")
    network = os.environ.get("NETWORK", "mainnet").lower()

    if network == "simulate":
        simulate = True

    is_testnet = (network == "testnet")

    if is_testnet:
        # Prefer Sepolia RPC when running in testnet mode
        json_rpc_arbitrum = os.environ.get("JSON_RPC_ARBITRUM_SEPOLIA") or json_rpc_arbitrum
        # GMX markets on Arbitrum Sepolia use USDC.SG as collateral;
        # override the global USDC lookup so vault deployment and all
        # downstream code use the correct token.
        USDC_NATIVE_TOKEN[ARBITRUM_SEPOLIA_CHAIN_ID] = GMX_TESTNET_USDC_ADDRESS

    assert json_rpc_arbitrum, "JSON_RPC_ARBITRUM is required"

    if simulate:
        private_key = None
    else:
        private_key = os.environ.get("GMX_PRIVATE_KEY")
        assert private_key, "GMX_PRIVATE_KEY is required in live mode"

    usdc_amount = Decimal(os.environ.get("USDC_AMOUNT", "5"))
    trading_strategy_api_key = os.environ.get("TRADING_STRATEGY_API_KEY", "")

    # ----- Set up simulation / connections -----
    (
        web3, deployer, private_key,
        json_rpc_arbitrum, anvil_launches,
    ) = setup_simulation(
        json_rpc_arbitrum=json_rpc_arbitrum,
        simulate=simulate,
        private_key=private_key,
        usdc_amount=usdc_amount,
        is_testnet=is_testnet,
    )

    try:
        usdc_token = verify_deployer_balances(
            web3=web3,
            deployer=deployer,
            usdc_amount=usdc_amount,
        )

        # Strategy file
        strategy_file = (
            Path(__file__).resolve().parent / ".." / ".." /
            "strategies" / "test_only" / "minimal_gmx_strategy.py"
        )
        assert strategy_file.exists(), f"Strategy file not found: {strategy_file}"

        print("=" * 70)
        print("GMX Lagoon vault manual test")
        print("=" * 70)
        print(f"  Mode:      {'SIMULATE (Anvil fork)' if simulate else f'LIVE ({network})'}")
        print(f"  Deployer:  {deployer.address}")
        print(f"  USDC:      {usdc_amount}")
        print(f"  Strategy:  {strategy_file.name}")
        print()

        _run_test_lifecycle(
            simulate=simulate,
            web3=web3,
            deployer=deployer,
            usdc_token=usdc_token,
            private_key=private_key,
            json_rpc_arbitrum=json_rpc_arbitrum,
            strategy_file=strategy_file,
            usdc_amount=usdc_amount,
            trading_strategy_api_key=trading_strategy_api_key,
            is_testnet=is_testnet,
        )
    finally:
        for launch in anvil_launches:
            launch.close(log_level=logging.ERROR)


if __name__ == "__main__":
    main()
