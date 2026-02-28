"""Manual test: Cross-chain Lagoon vault lifecycle via trade-executor CLI.

Exercises the full lifecycle of a multichain Lagoon vault on
Arbitrum Sepolia + Base Sepolia using trade-executor CLI commands:

1. Deploy multichain vault via ``lagoon-deploy-vault --strategy-file=...``
2. Initialise state via ``init``
3. Deposit USDC into the vault
4. Settle vault via ``lagoon-settle``
5. Run 5 strategy cycles via ``start`` with ``RUN_SINGLE_CYCLE=true``:
   - Cycle 1: Bridge USDC from Arbitrum Sepolia -> Base Sepolia via CCTP
   - (complete CCTP bridge: spoof on Anvil, wait for attestation on testnet)
   - Cycle 2: Swap USDC -> WETH on Base Sepolia via Uniswap v3
   - Cycle 3: Sell WETH -> USDC on Base Sepolia
   - Cycle 4: Bridge USDC from Base Sepolia -> Arbitrum Sepolia via reverse CCTP
   - (complete CCTP bridge: spoof on Anvil, wait for attestation on testnet)
   - Cycle 5: No-op, verify funds returned
6. Final settle + verification

The script uses the ``lagoon_crosschain_manual_test`` strategy module,
which creates a universe with CCTP bridge pairs and a WETH/USDC Uniswap v3
pair on Base Sepolia.

Modes
-----

**Simulated (Anvil forks)** — set ``SIMULATE=true``:

- Forks both testnets locally with Anvil
- Funds the deployer with ETH and USDC automatically
- Replaces CCTP attesters on both forks so attestations are forged
  instantly (no Circle Iris API polling)
- Fast — completes in minutes

**Live testnets** — omit ``SIMULATE`` or set to ``false``:

- Runs against real Arbitrum Sepolia + Base Sepolia
- Deployer must already hold testnet ETH + USDC
- CCTP attestations are polled from Circle's Iris API (15–20 min each)
- Requires Forge for from-scratch Lagoon deployment

Prerequisites
-------------

- Forge (for from-scratch Lagoon deployment on testnets)
- Testnet ETH on Arbitrum Sepolia and Base Sepolia
  (use LearnWeb3 and thirdweb faucets) — live mode only
- Testnet USDC on Arbitrum Sepolia
  (use Circle faucet: https://faucet.circle.com/) — live mode only
- A WETH/USDC Uniswap v3 pool on Base Sepolia (provide address via env var)

Environment variables
---------------------

``SIMULATE``
    Set to ``true`` to fork testnets with Anvil. Deployer is funded
    automatically and CCTP attestations are forged locally.
    Default: ``false`` (live testnet mode).

``JSON_RPC_ARBITRUM_SEPOLIA``
    Arbitrum Sepolia RPC URL. Required.

``JSON_RPC_BASE_SEPOLIA``
    Base Sepolia RPC URL. Required.

``LAGOON_MULTCHAIN_TEST_PRIVATE_KEY``
    Deployer private key. Only needed in live testnet mode (must hold
    testnet ETH + USDC). In simulate mode a random key is generated
    and the account is funded automatically.

``USDC_AMOUNT``
    Amount of USDC to deposit into the vault (default: ``10``).

``WETH_USDC_POOL_BASE_SEPOLIA``
    Uniswap v3 WETH/USDC pool address on Base Sepolia.
    Defaults to the well-known 0.05% fee tier pool.

``ATTESTATION_TIMEOUT``
    Maximum seconds to wait for CCTP attestation (default: ``3600``).
    Only used in live testnet mode. Testnet attestations can take
    15–20 minutes.

``TRADING_STRATEGY_API_KEY``
    TradingStrategy.ai API key. Optional for code-based strategies,
    but may be required by the ``start`` command.

Usage
-----

Simulated (Anvil forks):

.. code-block:: shell

    SIMULATE=true \\
    JSON_RPC_ARBITRUM_SEPOLIA="https://..." \\
    JSON_RPC_BASE_SEPOLIA="https://..." \\
    poetry run python scripts/lagoon/test-lagoon-crosschain-te.py

Live testnets:

.. code-block:: shell

    JSON_RPC_ARBITRUM_SEPOLIA="https://..." \\
    JSON_RPC_BASE_SEPOLIA="https://..." \\
    LAGOON_MULTCHAIN_TEST_PRIVATE_KEY="0x..." \\
    poetry run python scripts/lagoon/test-lagoon-crosschain-te.py
"""

import json
import logging
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest import mock

from eth_account.signers.local import LocalAccount

from eth_defi.cctp.attestation import wait_for_cctp_attestation
from eth_defi.cctp.receive import prepare_receive_message
from eth_defi.cctp.testing import replace_attester_on_fork, craft_cctp_message, forge_attestation
from eth_defi.erc_4626.vault_protocol.lagoon.testing import fund_lagoon_vault
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fork_network_anvil, set_balance, fund_erc20_on_anvil, AnvilLaunch
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.utils import setup_console_logging

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus

logger = logging.getLogger(__name__)

#: Arbitrum Sepolia chain ID
ARBITRUM_SEPOLIA_CHAIN_ID = 421614

#: Base Sepolia chain ID
BASE_SEPOLIA_CHAIN_ID = 84532


def run_cli(args: list[str], env: dict):
    """Run a trade-executor CLI command with the given environment.

    Uses ``mock.patch.dict`` to set environment variables cleanly,
    matching the pattern used by the trade-executor test suite.
    Preserves ``PATH`` and ``HOME`` so subprocesses (e.g. Forge) work.

    Clears the ``get_safe_cached_latest_block_number()`` block cache
    before each call so that vault reads use a fresh block number.
    Without this, newly deployed contracts appear empty because the
    cached block predates the deployment.
    """
    from eth_defi.provider.broken_provider import _latest_delayed_block_number_cache
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


def spoof_cctp_attestation(
    dest_web3,
    source_chain_id: int,
    dest_chain_id: int,
    mint_recipient: str,
    amount_raw: int,
    deployer: HotWallet,
    test_attester: LocalAccount,
    nonce: int = 999_999_000,
):
    """Forge a CCTP attestation and call receiveMessage on an Anvil fork.

    This function **only works on Anvil forks** where the CCTP attester has
    been replaced with a test account via :func:`replace_attester_on_fork`.
    It crafts a synthetic CCTP message and signs it with the test attester,
    then submits ``receiveMessage()`` on the destination fork.

    For real testnet or mainnet deployments, use :func:`complete_cctp_bridge`
    instead, which polls Circle's Iris API for a genuine attestation.

    :param dest_web3:
        Web3 connection to the destination chain (Anvil fork).

    :param source_chain_id:
        Numeric chain ID of the source chain.

    :param dest_chain_id:
        Numeric chain ID of the destination chain.

    :param mint_recipient:
        Address that will receive the minted USDC.

    :param amount_raw:
        Raw USDC amount (6 decimals).

    :param deployer:
        Hot wallet for signing the receiveMessage transaction.

    :param test_attester:
        Test attester account from :func:`replace_attester_on_fork`.

    :param nonce:
        Message nonce for the forged CCTP message.
    """
    from eth_defi.cctp.transfer import _resolve_cctp_domain

    source_domain = _resolve_cctp_domain(source_chain_id)
    dest_domain = _resolve_cctp_domain(dest_chain_id)

    logger.info(
        "Spoofing CCTP attestation (Anvil): src_chain=%d (domain %d), dest_chain=%d (domain %d), amount=%d, recipient=%s",
        source_chain_id, source_domain, dest_chain_id, dest_domain, amount_raw, mint_recipient,
    )

    message_bytes = craft_cctp_message(
        source_domain=source_domain,
        destination_domain=dest_domain,
        nonce=nonce,
        mint_recipient=mint_recipient,
        amount=amount_raw,
        burn_token=USDC_NATIVE_TOKEN[source_chain_id],
        testnet=True,
    )
    attestation_bytes = forge_attestation(message_bytes, test_attester)

    logger.info("Forged attestation, calling receiveMessage on chain %d", dest_chain_id)

    receive_fn = prepare_receive_message(dest_web3, message_bytes, attestation_bytes)

    # Force-reset nonce for the destination chain — the deployer may have
    # a higher nonce from the source chain, and sync_nonce() refuses to decrease.
    deployer.current_nonce = dest_web3.eth.get_transaction_count(deployer.address)
    tx_hash = deployer.transact_and_broadcast_with_contract(receive_fn)
    assert_transaction_success_with_explanation(dest_web3, tx_hash)

    logger.info("receiveMessage successful: %s", tx_hash.hex() if hasattr(tx_hash, "hex") else tx_hash)


def complete_cctp_bridge(
    dest_web3,
    source_chain_id: int,
    dest_chain_id: int,
    burn_tx_hash: str,
    mint_recipient: str,
    amount_raw: int,
    deployer: HotWallet,
    attestation_timeout: float = 3600.0,
):
    """Wait for a real CCTP attestation and call receiveMessage on the destination chain.

    Polls Circle's Iris API until the attestation is ready, then submits
    ``receiveMessage()``. Used for real testnet and mainnet deployments.
    Anyone can call ``receiveMessage()`` — it is permissionless.

    Testnet attestations typically take 15–20 minutes.

    :param dest_web3:
        Web3 connection to the destination chain.

    :param source_chain_id:
        Numeric chain ID of the source chain.

    :param dest_chain_id:
        Numeric chain ID of the destination chain.

    :param burn_tx_hash:
        Transaction hash of the depositForBurn call.

    :param mint_recipient:
        Address that will receive the minted USDC.

    :param amount_raw:
        Raw USDC amount (6 decimals).

    :param deployer:
        Hot wallet for signing the receiveMessage transaction.

    :param attestation_timeout:
        Maximum seconds to wait for attestation.
    """
    logger.info(
        "Waiting for CCTP attestation (Iris API): src_chain=%d, dest_chain=%d, tx=%s, amount=%d, recipient=%s",
        source_chain_id, dest_chain_id, burn_tx_hash, amount_raw, mint_recipient,
    )

    message_bytes, attestation_bytes = wait_for_cctp_attestation(
        source_chain_id=source_chain_id,
        dest_chain_id=dest_chain_id,
        burn_tx_hash=burn_tx_hash,
        dest_safe_address=mint_recipient,
        amount=amount_raw,
        simulate=False,
        timeout=attestation_timeout,
    )

    logger.info("Attestation received, calling receiveMessage on chain %d", dest_chain_id)

    receive_fn = prepare_receive_message(dest_web3, message_bytes, attestation_bytes)

    # Force-reset nonce for the destination chain — the deployer may have
    # a higher nonce from the source chain, and sync_nonce() refuses to decrease.
    deployer.current_nonce = dest_web3.eth.get_transaction_count(deployer.address)
    tx_hash = deployer.transact_and_broadcast_with_contract(receive_fn)
    assert_transaction_success_with_explanation(dest_web3, tx_hash)

    logger.info("receiveMessage successful: %s", tx_hash.hex() if hasattr(tx_hash, "hex") else tx_hash)


def main():
    setup_console_logging("info")

    # ----- Parse environment -----
    json_rpc_arb_sepolia = os.environ.get("JSON_RPC_ARBITRUM_SEPOLIA")
    json_rpc_base_sepolia = os.environ.get("JSON_RPC_BASE_SEPOLIA")
    simulate = os.environ.get("SIMULATE", "").lower() in ("true", "1")

    assert json_rpc_arb_sepolia, "JSON_RPC_ARBITRUM_SEPOLIA is required"
    assert json_rpc_base_sepolia, "JSON_RPC_BASE_SEPOLIA is required"

    if simulate:
        private_key = None  # Will create via HotWallet.create_for_testing() after Anvil forks
    else:
        private_key = os.environ.get("LAGOON_MULTCHAIN_TEST_PRIVATE_KEY")
        assert private_key, "LAGOON_MULTCHAIN_TEST_PRIVATE_KEY is required in live testnet mode"

    usdc_amount = Decimal(os.environ.get("USDC_AMOUNT", "10"))
    attestation_timeout = float(os.environ.get("ATTESTATION_TIMEOUT", "3600"))
    trading_strategy_api_key = os.environ.get("TRADING_STRATEGY_API_KEY", "")

    # ----- Set up Web3 connections -----
    anvil_launches: list[AnvilLaunch] = []
    test_attesters: dict[int, LocalAccount] = {}

    try:
        if simulate:
            # Fork the testnets with Anvil for local simulation
            logger.info("SIMULATE=true — forking testnets with Anvil")

            arb_launch = fork_network_anvil(json_rpc_arb_sepolia)
            anvil_launches.append(arb_launch)
            base_launch = fork_network_anvil(json_rpc_base_sepolia)
            anvil_launches.append(base_launch)

            arb_web3 = create_multi_provider_web3(arb_launch.json_rpc_url)
            base_web3 = create_multi_provider_web3(base_launch.json_rpc_url)

            # Override RPC URLs for CLI commands to point at Anvil forks
            json_rpc_arb_sepolia = arb_launch.json_rpc_url
            json_rpc_base_sepolia = base_launch.json_rpc_url

            # Create a random deployer wallet, funded with ETH on the primary fork
            deployer = HotWallet.create_for_testing(arb_web3, eth_amount=100)
            private_key = "0x" + deployer.account.key.hex()

            # Fund deployer with ETH on the second fork too
            set_balance(base_web3, deployer.address, 100 * 10**18)

            # Fund deployer with USDC on Arbitrum Sepolia fork
            usdc_raw = int(usdc_amount * 10**6) * 10  # 10x for headroom
            fund_erc20_on_anvil(arb_web3, USDC_NATIVE_TOKEN[ARBITRUM_SEPOLIA_CHAIN_ID], deployer.address, usdc_raw)

            logger.info("Deployer %s funded with 100 ETH + %d USDC on Arb Sepolia fork", deployer.address, usdc_raw // 10**6)

            # Replace CCTP attesters on both forks
            test_attesters[ARBITRUM_SEPOLIA_CHAIN_ID] = replace_attester_on_fork(arb_web3)
            test_attesters[BASE_SEPOLIA_CHAIN_ID] = replace_attester_on_fork(base_web3)
            logger.info("CCTP attesters replaced on both forks")
        else:
            arb_web3 = create_multi_provider_web3(json_rpc_arb_sepolia)
            base_web3 = create_multi_provider_web3(json_rpc_base_sepolia)
            deployer = HotWallet.from_private_key(private_key)

        assert arb_web3.eth.chain_id == ARBITRUM_SEPOLIA_CHAIN_ID, \
            f"Expected Arbitrum Sepolia ({ARBITRUM_SEPOLIA_CHAIN_ID}), got {arb_web3.eth.chain_id}"
        assert base_web3.eth.chain_id == BASE_SEPOLIA_CHAIN_ID, \
            f"Expected Base Sepolia ({BASE_SEPOLIA_CHAIN_ID}), got {base_web3.eth.chain_id}"

        deployer.sync_nonce(arb_web3)

        # Verify deployer has gas
        arb_balance = arb_web3.eth.get_balance(deployer.address)
        base_balance = base_web3.eth.get_balance(deployer.address)
        logger.info("Deployer: %s", deployer.address)
        logger.info("  Arbitrum Sepolia ETH: %.6f", arb_balance / 10**18)
        logger.info("  Base Sepolia ETH:     %.6f", base_balance / 10**18)
        assert arb_balance > 0, f"Deployer has no ETH on Arbitrum Sepolia. Fund {deployer.address} first."
        assert base_balance > 0, f"Deployer has no ETH on Base Sepolia. Fund {deployer.address} first."

        # Verify deployer has USDC
        arb_usdc = fetch_erc20_details(arb_web3, USDC_NATIVE_TOKEN[ARBITRUM_SEPOLIA_CHAIN_ID])
        deployer_usdc = arb_usdc.fetch_balance_of(deployer.address)
        logger.info("  Arbitrum Sepolia USDC: %s", deployer_usdc)
        assert deployer_usdc >= usdc_amount, \
            f"Deployer needs {usdc_amount} USDC on Arbitrum Sepolia but has {deployer_usdc}. " \
            f"Get testnet USDC from https://faucet.circle.com/"

        # Strategy file
        strategy_file = (
            Path(__file__).resolve().parent / ".." / ".." /
            "strategies" / "test_only" / "lagoon_crosschain_manual_test.py"
        )
        assert strategy_file.exists(), f"Strategy file not found: {strategy_file}"

        print("=" * 70)
        print("Cross-chain Lagoon vault manual test")
        print("=" * 70)
        print(f"  Mode:           {'SIMULATE (Anvil forks)' if simulate else 'LIVE (real testnets)'}")
        print(f"  Deployer:       {deployer.address}")
        print(f"  USDC deposit:   {usdc_amount}")
        print(f"  Strategy:       {strategy_file.name}")
        if not simulate:
            print(f"  Attest timeout: {attestation_timeout}s")
        print()

        _run_test_lifecycle(
            simulate=simulate,
            test_attesters=test_attesters,
            arb_web3=arb_web3,
            base_web3=base_web3,
            deployer=deployer,
            arb_usdc=arb_usdc,
            private_key=private_key,
            json_rpc_arb_sepolia=json_rpc_arb_sepolia,
            json_rpc_base_sepolia=json_rpc_base_sepolia,
            strategy_file=strategy_file,
            usdc_amount=usdc_amount,
            trading_strategy_api_key=trading_strategy_api_key,
            attestation_timeout=attestation_timeout,
        )
    finally:
        for launch in anvil_launches:
            launch.close(log_level=logging.ERROR)


def _run_test_lifecycle(
    *,
    simulate: bool,
    test_attesters: dict[int, LocalAccount],
    arb_web3,
    base_web3,
    deployer: HotWallet,
    arb_usdc,
    private_key: str,
    json_rpc_arb_sepolia: str,
    json_rpc_base_sepolia: str,
    strategy_file: Path,
    usdc_amount: Decimal,
    trading_strategy_api_key: str,
    attestation_timeout: float,
):
    """Run the full test lifecycle (extracted to avoid deep nesting)."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        state_file = str(Path(tmp_dir) / "state.json")
        vault_record_file = str(Path(tmp_dir) / "vault-record.txt")
        cache_path = str(Path(tmp_dir) / "cache")

        # ===================================================================
        # Step 1: Deploy multichain vault
        # ===================================================================
        print("\n=== Step 1: Deploy multichain vault ===")

        deploy_env = {
            "STRATEGY_FILE": str(strategy_file),
            "PRIVATE_KEY": private_key,
            "JSON_RPC_ARBITRUM_SEPOLIA": json_rpc_arb_sepolia,
            "JSON_RPC_BASE_SEPOLIA": json_rpc_base_sepolia,
            "VAULT_RECORD_FILE": vault_record_file,
            "FUND_NAME": "Test Crosschain Vault",
            "FUND_SYMBOL": "TCV",
            "ANY_ASSET": "true",
            "UNIT_TESTING": "true",
            "LOG_LEVEL": "info",
        }

        run_cli(["lagoon-deploy-vault"], deploy_env)

        # Read deployment record
        deployment_json = vault_record_file.replace(".txt", ".json")
        with open(deployment_json) as f:
            deployment_data = json.load(f)

        arb_dep = deployment_data["deployments"]["arbitrum_sepolia"]
        base_dep = deployment_data["deployments"]["base_sepolia"]
        vault_address = arb_dep["vault_address"]
        safe_address = arb_dep["safe_address"]
        arb_module = arb_dep["module_address"]
        base_module = base_dep["module_address"]

        print(f"  Vault:      {vault_address}")
        print(f"  Safe:       {safe_address}")
        print(f"  Arb module: {arb_module}")
        print(f"  Base module: {base_module}")

        # ===================================================================
        # Step 2: Initialise state
        # ===================================================================
        print("\n=== Step 2: Initialise state ===")

        base_env = {
            "ID": "test-crosschain",
            "STRATEGY_FILE": str(strategy_file),
            "STATE_FILE": state_file,
            "PRIVATE_KEY": private_key,
            "JSON_RPC_ARBITRUM_SEPOLIA": json_rpc_arb_sepolia,
            "JSON_RPC_BASE_SEPOLIA": json_rpc_base_sepolia,
            "ASSET_MANAGEMENT_MODE": "lagoon",
            "VAULT_ADDRESS": vault_address,
            "VAULT_ADAPTER_ADDRESS": arb_module,
            "UNIT_TESTING": "true",
            "LOG_LEVEL": "info",
            "CACHE_PATH": cache_path,
            "SATELLITE_MODULES": json.dumps({"base_sepolia": base_module}),
            "MIN_GAS_BALANCE": "0.0",
        }
        if trading_strategy_api_key:
            base_env["TRADING_STRATEGY_API_KEY"] = trading_strategy_api_key

        run_cli(["init"], {**base_env, "NAME": "Test Crosschain Vault"})

        assert Path(state_file).exists(), "State file was not created"
        print(f"  State file: {state_file}")

        if not simulate:
            # Allow testnet RPC state to propagate after vault deployment
            import time as _time
            logger.info("Waiting 30s for testnet state propagation after init")
            _time.sleep(30)

        # ===================================================================
        # Step 3: Deposit USDC into the vault
        # ===================================================================
        print("\n=== Step 3: Deposit USDC ===")

        deployer.sync_nonce(arb_web3)
        fund_lagoon_vault(
            web3=arb_web3,
            vault_address=vault_address,
            asset_manager=deployer.address,
            test_account_with_balance=deployer.address,
            trading_strategy_module_address=arb_module,
            amount=usdc_amount,
            hot_wallet=deployer,
        )

        safe_usdc = arb_usdc.fetch_balance_of(safe_address)
        print(f"  Safe USDC after deposit: {safe_usdc}")

        # ===================================================================
        # Step 4: Settle vault (process deposit, update NAV)
        # ===================================================================
        print("\n=== Step 4: Settle vault ===")

        settle_env = {k: v for k, v in base_env.items() if k != "NAME"}
        settle_env["SYNC_INTEREST"] = "false"

        # Testnet RPCs can be slow to propagate new contract state.
        # Retry the settle step with increasing delays.
        # The block cache is cleared in run_cli() before each call.
        import time as _time

        for attempt in range(1, 6):
            if not simulate:
                delay = 30 * attempt
                logger.info("Waiting %ds for testnet state propagation (attempt %d/5)", delay, attempt)
                _time.sleep(delay)

            try:
                run_cli(["lagoon-settle"], settle_env)
                break
            except Exception as e:
                if attempt == 5:
                    raise
                logger.warning("Settle attempt %d failed: %s. Retrying...", attempt, e)

        print("  Vault settled")

        # ===================================================================
        # Step 5: Cycle 1 — Bridge USDC from Arbitrum Sepolia to Base Sepolia
        # ===================================================================
        print("\n=== Step 5: Cycle 1 — Bridge USDC to Base Sepolia ===")

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
        assert len(state.portfolio.open_positions) >= 1, \
            f"Expected at least 1 open position after cycle 1, got {len(state.portfolio.open_positions)}"

        # Find the bridge position
        bridge_positions = [
            pos for pos in state.portfolio.open_positions.values()
            if pos.pair.is_cctp_bridge()
        ]
        assert len(bridge_positions) == 1, \
            f"Expected 1 bridge position after cycle 1, got {len(bridge_positions)}"

        bridge_trade = list(bridge_positions[0].trades.values())[0]
        assert bridge_trade.get_status() == TradeStatus.success, \
            f"Bridge trade status: {bridge_trade.get_status()}"

        burn_tx_hash = bridge_trade.blockchain_transactions[-1].tx_hash
        bridge_amount_raw = int(bridge_trade.planned_reserve * 10**6)
        print(f"  Bridge trade: {bridge_trade.get_status()}")
        print(f"  Burn TX: {burn_tx_hash}")
        print(f"  Amount: {bridge_trade.planned_reserve} USDC")

        # ===================================================================
        # Step 5b: CCTP attestation + receive on Base Sepolia
        # ===================================================================
        print("\n=== Step 5b: CCTP attestation + receive on Base Sepolia ===")

        if simulate:
            spoof_cctp_attestation(
                dest_web3=base_web3,
                source_chain_id=ARBITRUM_SEPOLIA_CHAIN_ID,
                dest_chain_id=BASE_SEPOLIA_CHAIN_ID,
                mint_recipient=safe_address,
                amount_raw=bridge_amount_raw,
                deployer=deployer,
                test_attester=test_attesters[BASE_SEPOLIA_CHAIN_ID],
            )
        else:
            complete_cctp_bridge(
                dest_web3=base_web3,
                source_chain_id=ARBITRUM_SEPOLIA_CHAIN_ID,
                dest_chain_id=BASE_SEPOLIA_CHAIN_ID,
                burn_tx_hash=burn_tx_hash,
                mint_recipient=safe_address,
                amount_raw=bridge_amount_raw,
                deployer=deployer,
                attestation_timeout=attestation_timeout,
            )

        # Verify USDC arrived on Base Sepolia (with retry for RPC propagation)
        base_usdc = fetch_erc20_details(base_web3, USDC_NATIVE_TOKEN[BASE_SEPOLIA_CHAIN_ID])
        for _check in range(6):
            base_safe_usdc = base_usdc.fetch_balance_of(safe_address)
            if base_safe_usdc > 0:
                break
            if not simulate:
                logger.info("Base Safe USDC still 0, waiting 10s for RPC propagation (attempt %d/6)", _check + 1)
                _time.sleep(10)
        print(f"  Base Safe USDC after bridge: {base_safe_usdc}")
        assert base_safe_usdc > 0, "USDC did not arrive on Base Sepolia"

        # ===================================================================
        # Step 6: Cycle 2 — Swap USDC -> WETH on Base Sepolia
        # ===================================================================
        print("\n=== Step 6: Cycle 2 — Swap USDC -> WETH ===")

        run_cli(["start"], start_env)

        state = load_state(state_file)
        weth_positions = [
            pos for pos in state.portfolio.open_positions.values()
            if pos.pair.base.token_symbol == "WETH"
        ]
        assert len(weth_positions) == 1, \
            f"Expected 1 WETH position after cycle 2, got {len(weth_positions)}"

        weth_trade = list(weth_positions[0].trades.values())[0]
        assert weth_trade.get_status() == TradeStatus.success, \
            f"WETH buy trade status: {weth_trade.get_status()}"

        print(f"  WETH buy trade: {weth_trade.get_status()}")
        print(f"  WETH acquired: {weth_trade.executed_quantity}")

        # ===================================================================
        # Step 7: Cycle 3 — Sell WETH -> USDC on Base Sepolia
        # ===================================================================
        print("\n=== Step 7: Cycle 3 — Sell WETH -> USDC ===")

        run_cli(["start"], start_env)

        state = load_state(state_file)
        weth_closed = [
            pos for pos in state.portfolio.closed_positions.values()
            if pos.pair.base.token_symbol == "WETH"
        ]
        assert len(weth_closed) == 1, \
            f"Expected 1 closed WETH position after cycle 3, got {len(weth_closed)}"

        weth_sell_trade = list(weth_closed[0].trades.values())[-1]
        assert weth_sell_trade.get_status() == TradeStatus.success, \
            f"WETH sell trade status: {weth_sell_trade.get_status()}"

        print(f"  WETH sell trade: {weth_sell_trade.get_status()}")

        # ===================================================================
        # Step 8: Cycle 4 — Bridge USDC from Base Sepolia back to Arb Sepolia
        # ===================================================================
        print("\n=== Step 8: Cycle 4 — Bridge USDC back to Arbitrum Sepolia ===")

        run_cli(["start"], start_env)

        state = load_state(state_file)
        reverse_bridge_positions = [
            pos for pos in state.portfolio.open_positions.values()
            if pos.pair.is_cctp_bridge() and pos.pair.quote.chain_id == BASE_SEPOLIA_CHAIN_ID
        ]
        assert len(reverse_bridge_positions) == 1, \
            f"Expected 1 reverse bridge position after cycle 4, got {len(reverse_bridge_positions)}"

        reverse_trade = list(reverse_bridge_positions[0].trades.values())[0]
        assert reverse_trade.get_status() == TradeStatus.success, \
            f"Reverse bridge trade status: {reverse_trade.get_status()}"

        reverse_burn_tx = reverse_trade.blockchain_transactions[-1].tx_hash
        reverse_amount_raw = int(reverse_trade.planned_reserve * 10**6)
        print(f"  Reverse bridge trade: {reverse_trade.get_status()}")
        print(f"  Burn TX: {reverse_burn_tx}")
        print(f"  Amount: {reverse_trade.planned_reserve} USDC")

        # ===================================================================
        # Step 8b: CCTP attestation + receive on Arbitrum Sepolia
        # ===================================================================
        print("\n=== Step 8b: CCTP attestation + receive on Arbitrum Sepolia ===")

        if simulate:
            spoof_cctp_attestation(
                dest_web3=arb_web3,
                source_chain_id=BASE_SEPOLIA_CHAIN_ID,
                dest_chain_id=ARBITRUM_SEPOLIA_CHAIN_ID,
                mint_recipient=safe_address,
                amount_raw=reverse_amount_raw,
                deployer=deployer,
                test_attester=test_attesters[ARBITRUM_SEPOLIA_CHAIN_ID],
                nonce=999_999_001,
            )
        else:
            complete_cctp_bridge(
                dest_web3=arb_web3,
                source_chain_id=BASE_SEPOLIA_CHAIN_ID,
                dest_chain_id=ARBITRUM_SEPOLIA_CHAIN_ID,
                burn_tx_hash=reverse_burn_tx,
                mint_recipient=safe_address,
                amount_raw=reverse_amount_raw,
                deployer=deployer,
                attestation_timeout=attestation_timeout,
            )

        # Verify USDC returned to Arbitrum Sepolia (with retry for RPC propagation)
        for _check in range(6):
            arb_safe_usdc_after = arb_usdc.fetch_balance_of(safe_address)
            if arb_safe_usdc_after > 0:
                break
            if not simulate:
                logger.info("Arb Safe USDC still 0, waiting 10s for RPC propagation (attempt %d/6)", _check + 1)
                _time.sleep(10)
        print(f"  Arb Safe USDC after reverse bridge: {arb_safe_usdc_after}")

        # ===================================================================
        # Step 9: Cycle 5 — No-op (verify funds returned)
        # ===================================================================
        print("\n=== Step 9: Cycle 5 — No-op (verify funds returned) ===")

        run_cli(["start"], start_env)

        state = load_state(state_file)
        final_equity = state.portfolio.get_total_equity()
        print(f"  Final equity: {final_equity}")

        # ===================================================================
        # Step 10: Final settle + verification
        # ===================================================================
        print("\n=== Step 10: Final settle + verification ===")

        run_cli(["lagoon-settle"], settle_env)

        # Check balances
        final_arb_safe_usdc = arb_usdc.fetch_balance_of(safe_address)
        final_base_safe_usdc = base_usdc.fetch_balance_of(safe_address)

        print()
        print("=" * 70)
        print("Final status")
        print("=" * 70)
        print(f"  Arb Safe USDC:  {final_arb_safe_usdc}")
        print(f"  Base Safe USDC: {final_base_safe_usdc}")
        print(f"  Portfolio equity: {final_equity}")
        print(f"  Open positions:  {len(state.portfolio.open_positions)}")
        print(f"  Closed positions: {len(state.portfolio.closed_positions)}")

        # Verify the round trip was successful
        # We expect most USDC to be back on Arbitrum, minus swap fees
        assert final_arb_safe_usdc > 0, "No USDC returned to Arbitrum Sepolia Safe"

        print()
        print("All ok!")


if __name__ == "__main__":
    main()
