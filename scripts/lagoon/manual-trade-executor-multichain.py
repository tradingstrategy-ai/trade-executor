"""Manual test: Cross-chain Lagoon vault lifecycle via trade-executor CLI.

Exercises the full lifecycle of a multichain Lagoon vault on
Arbitrum/Base mainnet or Arbitrum Sepolia/Base Sepolia using
trade-executor CLI commands:

1. Deploy multichain vault via ``lagoon-deploy-vault --strategy-file=...``
2. Initialise state via ``init``
3. Deposit USDC into the vault
4. Settle vault via ``lagoon-settle``
5. Run 5 strategy cycles via ``start`` with ``RUN_SINGLE_CYCLE=true``:
   - Cycle 1: Bridge USDC from Arbitrum -> Base via CCTP
   - (complete CCTP bridge: spoof on Anvil, or wait for a real attestation)
   - Cycle 2: Swap USDC -> WETH on Base via Uniswap v3
   - Cycle 3: Sell WETH -> USDC on Base
   - Cycle 4: Bridge USDC from Base -> Arbitrum via reverse CCTP
   - (complete CCTP bridge: spoof on Anvil, or wait for a real attestation)
   - Cycle 5: No-op, verify funds returned
6. Final settle + verification

The script chooses a matching strategy module for the selected network,
creating a universe with CCTP bridge pairs and a WETH/USDC Uniswap v3
pair on Base or Base Sepolia.

See lagoon-multichain.rst for more details on multichain bridge flows.

Modes
-----

**Simulated (Anvil forks)** — set ``SIMULATE=true``:

- Forks both mainnets locally with Anvil
- Funds the deployer with ETH and USDC automatically
- Replaces CCTP attesters on both forks so attestations are forged
  instantly (no Circle Iris API polling)
- Fast — completes in minutes

**Live network** — omit ``SIMULATE`` or set to ``false``:

- Runs against real Arbitrum + Base or Arbitrum Sepolia + Base Sepolia
- Deployer must already hold gas tokens + USDC on the selected network
- CCTP attestations are polled from Circle's Iris API
- Requires Forge for from-scratch Lagoon deployment

Prerequisites
-------------

- Forge (for from-scratch Lagoon deployment)
- Gas on Arbitrum and Base for mainnet mode, or on Arbitrum Sepolia and
  Base Sepolia for testnet mode
- USDC on the selected Arbitrum network
- A WETH/USDC Uniswap v3 pool on the selected Base network

Environment variables
---------------------

``SIMULATE``
    Set to ``true`` to fork the selected network pair with Anvil. Deployer is funded
    automatically and CCTP attestations are forged locally.
    Default: ``false``.

``LAGOON_CROSSCHAIN_NETWORK``
    Optional network selector: ``mainnet`` or ``testnet``.
    If omitted, mainnet is used when ``JSON_RPC_ARBITRUM`` and
    ``JSON_RPC_BASE`` are present. Otherwise Sepolia is used when
    ``JSON_RPC_ARBITRUM_SEPOLIA`` and ``JSON_RPC_BASE_SEPOLIA`` are present.

``JSON_RPC_ARBITRUM``
    Arbitrum mainnet RPC URL. Required for mainnet mode.

``JSON_RPC_BASE``
    Base mainnet RPC URL. Required for mainnet mode.

``JSON_RPC_ARBITRUM_SEPOLIA``
    Arbitrum Sepolia RPC URL. Required for testnet mode.

``JSON_RPC_BASE_SEPOLIA``
    Base Sepolia RPC URL. Required for testnet mode.

``LAGOON_MULTCHAIN_TEST_PRIVATE_KEY``
    Deployer private key. Only needed in live mode (must hold
    gas tokens + USDC). In simulate mode a random key is generated
    and the account is funded automatically.

``USDC_AMOUNT``
    Amount of USDC to deposit into the vault (default: ``10``).

``WETH_USDC_POOL_BASE``
    Uniswap v3 WETH/USDC pool address on Base.
    Defaults to the canonical 0.05% fee tier pool.

``WETH_USDC_POOL_BASE_SEPOLIA``
    Uniswap v3 WETH/USDC pool address on Base Sepolia.
    Defaults to the well-known 0.05% fee tier pool.

``ATTESTATION_TIMEOUT``
    Maximum seconds to wait for CCTP attestation (default: ``3600``).
    Only used in live mode.

``TRADING_STRATEGY_API_KEY``
    TradingStrategy.ai API key. Optional for code-based strategies,
    but may be required by the ``start`` command.

Usage
-----

Simulated (Anvil forks):

.. code-block:: shell

    SIMULATE=true \\
    LAGOON_CROSSCHAIN_NETWORK=mainnet \\
    JSON_RPC_ARBITRUM="https://..." \\
    JSON_RPC_BASE="https://..." \\
    poetry run python scripts/lagoon/manual-trade-executor-multichain.py

Mainnet:

.. code-block:: shell

    LAGOON_CROSSCHAIN_NETWORK=mainnet \\
    JSON_RPC_ARBITRUM="https://..." \\
    JSON_RPC_BASE="https://..." \\
    LAGOON_MULTCHAIN_TEST_PRIVATE_KEY="0x..." \\
    poetry run python scripts/lagoon/manual-trade-executor-multichain.py

Testnet:

.. code-block:: shell

    LAGOON_CROSSCHAIN_NETWORK=testnet \\
    JSON_RPC_ARBITRUM_SEPOLIA="https://..." \\
    JSON_RPC_BASE_SEPOLIA="https://..." \\
    LAGOON_MULTCHAIN_TEST_PRIVATE_KEY="0x..." \\
    poetry run python scripts/lagoon/manual-trade-executor-multichain.py

If the script aborts before redeeming vault shares, USDC may remain
locked inside the deployed Gnosis Safe. Use ``recover-safe-usdc.py``
(in the same directory) to transfer that USDC back to the deployer.
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
from eth_defi.cctp.testing import (craft_cctp_message, forge_attestation,
                                   replace_attester_on_fork)
from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.erc_4626.vault_protocol.lagoon.testing import (
    fund_lagoon_vault, redeem_vault_shares)
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import (AnvilLaunch, fork_network_anvil,
                                     fund_erc20_on_anvil, set_balance)
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.utils import setup_console_logging

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus

logger = logging.getLogger(__name__)

#: Arbitrum mainnet chain ID
ARBITRUM_CHAIN_ID = 42161

#: Base mainnet chain ID
BASE_CHAIN_ID = 8453

#: Arbitrum Sepolia chain ID
ARBITRUM_SEPOLIA_CHAIN_ID = 421614

#: Base Sepolia chain ID
BASE_SEPOLIA_CHAIN_ID = 84532


def resolve_network_configuration() -> dict:
    """Resolve whether the script runs against mainnet or testnet."""
    requested_network = os.environ.get("LAGOON_CROSSCHAIN_NETWORK")

    if requested_network not in (None, "mainnet", "testnet"):
        raise AssertionError(
            f"LAGOON_CROSSCHAIN_NETWORK must be 'mainnet' or 'testnet', got {requested_network!r}",
        )

    mainnet_ready = bool(os.environ.get("JSON_RPC_ARBITRUM") and os.environ.get("JSON_RPC_BASE"))
    testnet_ready = bool(os.environ.get("JSON_RPC_ARBITRUM_SEPOLIA") and os.environ.get("JSON_RPC_BASE_SEPOLIA"))

    if requested_network is None:
        if mainnet_ready:
            requested_network = "mainnet"
        elif testnet_ready:
            requested_network = "testnet"
        else:
            raise AssertionError(
                "Set either JSON_RPC_ARBITRUM + JSON_RPC_BASE or "
                "JSON_RPC_ARBITRUM_SEPOLIA + JSON_RPC_BASE_SEPOLIA",
            )

    if requested_network == "mainnet":
        assert mainnet_ready, "JSON_RPC_ARBITRUM and JSON_RPC_BASE are required for mainnet mode"
        return {
            "network": "mainnet",
            "is_testnet": False,
            "json_rpc_arbitrum": os.environ["JSON_RPC_ARBITRUM"],
            "json_rpc_base": os.environ["JSON_RPC_BASE"],
            "rpc_env_keys": ("JSON_RPC_ARBITRUM", "JSON_RPC_BASE"),
            "chain_ids": (ARBITRUM_CHAIN_ID, BASE_CHAIN_ID),
            "chain_names": ("Arbitrum", "Base"),
            "chain_slugs": ("arbitrum", "base"),
            "strategy_file": Path(__file__).resolve().parent / ".." / ".." / "strategies" / "test_only" / "lagoon_crosschain_manual_test.py",
        }

    assert testnet_ready, "JSON_RPC_ARBITRUM_SEPOLIA and JSON_RPC_BASE_SEPOLIA are required for testnet mode"
    return {
        "network": "testnet",
        "is_testnet": True,
        "json_rpc_arbitrum": os.environ["JSON_RPC_ARBITRUM_SEPOLIA"],
        "json_rpc_base": os.environ["JSON_RPC_BASE_SEPOLIA"],
        "rpc_env_keys": ("JSON_RPC_ARBITRUM_SEPOLIA", "JSON_RPC_BASE_SEPOLIA"),
        "chain_ids": (ARBITRUM_SEPOLIA_CHAIN_ID, BASE_SEPOLIA_CHAIN_ID),
        "chain_names": ("Arbitrum Sepolia", "Base Sepolia"),
        "chain_slugs": ("arbitrum_sepolia", "base_sepolia"),
        "strategy_file": Path(__file__).resolve().parent / ".." / ".." / "strategies" / "test_only" / "lagoon_crosschain_manual_testnet.py",
    }


def _check_shares(web3, vault_address: str, deployer_address: str, label: str):
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

    Uses ``mock.patch.dict`` to set environment variables cleanly,
    matching the pattern used by the trade-executor test suite.
    Preserves ``PATH`` and ``HOME`` so subprocesses (e.g. Forge) work.

    Clears the ``get_safe_cached_latest_block_number()`` block cache
    before each call so that vault reads use a fresh block number.
    Without this, newly deployed contracts appear empty because the
    cached block predates the deployment.
    """
    from eth_defi.provider.broken_provider import \
        _latest_delayed_block_number_cache
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
    is_testnet: bool,
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
        testnet=is_testnet,
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


def wait_for_token_balance(
    token,
    address: str,
    *,
    simulate: bool,
    label: str = "Token",
    retries: int = 6,
    delay: float = 10.0,
) -> Decimal:
    """Wait for a non-zero token balance with retries for RPC propagation.

    In simulate mode (Anvil forks) the balance is expected immediately,
    so no delay is applied between checks. On live networks a short
    delay is inserted between retries to allow the RPC to catch up.

    :param token:
        ERC-20 token details (from :func:`fetch_erc20_details`).
    :param address:
        Address to check the balance of.
    :param simulate:
        Whether running against Anvil forks.
    :param label:
        Human-readable label for log messages.
    :param retries:
        Maximum number of check attempts.
    :param delay:
        Seconds to wait between retries (live mode only).
    :return:
        The token balance (asserts it is positive).
    """
    import time as _time

    for attempt in range(retries):
        balance = token.fetch_balance_of(address)
        if balance > 0:
            break
        if not simulate:
            logger.info(
                "%s balance still 0, waiting %.0fs for RPC propagation (attempt %d/%d)",
                label, delay, attempt + 1, retries,
            )
            _time.sleep(delay)

    assert balance > 0, f"{label} balance is 0 at {address}"
    return balance


def setup_simulation(
    *,
    network_config: dict,
    simulate: bool,
    private_key: str | None,
    usdc_amount: Decimal,
) -> tuple:
    """Set up Web3 connections, deployer wallet, and optionally Anvil forks.

    In simulate mode:

    - Forks both configured networks locally with Anvil
    - Creates Web3 connections pointing at Anvil forks
    - Creates a deployer wallet funded with ETH on both forks
    - Funds deployer with USDC on the configured Arbitrum fork
    - Replaces CCTP attesters on both forks

    In live mode:

    - Creates Web3 connections to real RPCs
    - Loads deployer from *private_key*

    :return:
        Tuple of ``(arb_web3, base_web3, deployer, private_key,
        json_rpc_arbitrum, json_rpc_base,
        test_attesters, anvil_launches)``.
    """
    json_rpc_arbitrum = network_config["json_rpc_arbitrum"]
    json_rpc_base = network_config["json_rpc_base"]
    source_chain_id, dest_chain_id = network_config["chain_ids"]
    source_chain_name, _ = network_config["chain_names"]

    anvil_launches: list[AnvilLaunch] = []
    test_attesters: dict[int, LocalAccount] = {}

    if simulate:
        logger.info("SIMULATE=true — forking mainnets with Anvil")

        arb_launch = fork_network_anvil(json_rpc_arbitrum)
        anvil_launches.append(arb_launch)
        base_launch = fork_network_anvil(json_rpc_base)
        anvil_launches.append(base_launch)

        arb_web3 = create_multi_provider_web3(arb_launch.json_rpc_url)
        base_web3 = create_multi_provider_web3(base_launch.json_rpc_url)

        # Override RPC URLs for CLI commands to point at Anvil forks
        json_rpc_arbitrum = arb_launch.json_rpc_url
        json_rpc_base = base_launch.json_rpc_url

        # Create a random deployer wallet, funded with ETH on the primary fork
        deployer = HotWallet.create_for_testing(arb_web3, eth_amount=100)
        private_key = "0x" + deployer.account.key.hex()

        # Fund deployer with ETH on the second fork too
        set_balance(base_web3, deployer.address, 100 * 10**18)

        # Fund deployer with USDC on Arbitrum fork
        arb_usdc_token = fetch_erc20_details(arb_web3, USDC_NATIVE_TOKEN[source_chain_id])
        usdc_raw = arb_usdc_token.convert_to_raw(usdc_amount) * 10  # 10x for headroom
        fund_erc20_on_anvil(arb_web3, USDC_NATIVE_TOKEN[source_chain_id], deployer.address, usdc_raw)

        logger.info(
            "Deployer %s funded with 100 ETH + %d USDC on %s fork",
            deployer.address,
            usdc_raw // 10**arb_usdc_token.decimals,
            source_chain_name,
        )

        # Replace CCTP attesters on both forks
        test_attesters[source_chain_id] = replace_attester_on_fork(arb_web3)
        test_attesters[dest_chain_id] = replace_attester_on_fork(base_web3)
        logger.info("CCTP attesters replaced on both forks")
    else:
        arb_web3 = create_multi_provider_web3(json_rpc_arbitrum)
        base_web3 = create_multi_provider_web3(json_rpc_base)
        deployer = HotWallet.from_private_key(private_key)

    assert arb_web3.eth.chain_id == source_chain_id, \
        f"Expected {source_chain_name} ({source_chain_id}), got {arb_web3.eth.chain_id}"
    assert base_web3.eth.chain_id == dest_chain_id, \
        f"Expected {network_config['chain_names'][1]} ({dest_chain_id}), got {base_web3.eth.chain_id}"

    deployer.sync_nonce(arb_web3)

    return (
        arb_web3, base_web3, deployer, private_key,
        json_rpc_arbitrum, json_rpc_base,
        test_attesters, anvil_launches,
    )


def verify_deployer_balances(
    *,
    network_config: dict,
    arb_web3,
    base_web3,
    deployer: HotWallet,
    usdc_amount: Decimal,
):
    """Verify deployer has ETH on both chains and sufficient USDC on the source chain.

    :return:
        The ``arb_usdc`` token details (from :func:`fetch_erc20_details`).
    """
    source_chain_id, _ = network_config["chain_ids"]
    source_chain_name, dest_chain_name = network_config["chain_names"]
    arb_balance = arb_web3.eth.get_balance(deployer.address)
    base_balance = base_web3.eth.get_balance(deployer.address)
    logger.info("Deployer: %s", deployer.address)
    logger.info("  %s ETH: %.6f", source_chain_name, arb_balance / 10**18)
    logger.info("  %s ETH: %.6f", dest_chain_name, base_balance / 10**18)
    assert arb_balance > 0, f"Deployer has no ETH on {source_chain_name}. Fund {deployer.address} first."
    assert base_balance > 0, f"Deployer has no ETH on {dest_chain_name}. Fund {deployer.address} first."

    arb_usdc = fetch_erc20_details(arb_web3, USDC_NATIVE_TOKEN[source_chain_id])
    deployer_usdc = arb_usdc.fetch_balance_of(deployer.address)
    logger.info("  %s USDC: %s", source_chain_name, deployer_usdc)
    assert deployer_usdc >= usdc_amount, \
        f"Deployer needs {usdc_amount} USDC on {source_chain_name} but has {deployer_usdc}."

    return arb_usdc


def main():
    setup_console_logging("warning")

    # ----- Parse environment -----
    network_config = resolve_network_configuration()
    json_rpc_arbitrum = network_config["json_rpc_arbitrum"]
    json_rpc_base = network_config["json_rpc_base"]
    simulate = os.environ.get("SIMULATE", "").lower() in ("true", "1")

    if simulate:
        private_key = None  # Will create via HotWallet.create_for_testing() after Anvil forks
    else:
        private_key = os.environ.get("LAGOON_MULTCHAIN_TEST_PRIVATE_KEY")
        assert private_key, "LAGOON_MULTCHAIN_TEST_PRIVATE_KEY is required in live mode"

    usdc_amount = Decimal(os.environ.get("USDC_AMOUNT", "10"))
    bridge_amount = os.environ.get("BRIDGE_AMOUNT", "3")
    swap_amount = os.environ.get("SWAP_AMOUNT", "2")
    reverse_bridge_amount = os.environ.get("REVERSE_BRIDGE_AMOUNT", "1")
    attestation_timeout = float(os.environ.get("ATTESTATION_TIMEOUT", "3600"))
    trading_strategy_api_key = os.environ.get("TRADING_STRATEGY_API_KEY", "")

    # ----- Set up simulation / connections -----
    (
        arb_web3, base_web3, deployer, private_key,
        json_rpc_arbitrum, json_rpc_base,
        test_attesters, anvil_launches,
    ) = setup_simulation(
        network_config=network_config,
        simulate=simulate,
        private_key=private_key,
        usdc_amount=usdc_amount,
    )

    try:
        arb_usdc = verify_deployer_balances(
            network_config=network_config,
            arb_web3=arb_web3,
            base_web3=base_web3,
            deployer=deployer,
            usdc_amount=usdc_amount,
        )

        strategy_file_override = os.environ.get("STRATEGY_FILE")
        if strategy_file_override:
            strategy_file = Path(strategy_file_override)
        else:
            strategy_file = network_config["strategy_file"]
        assert strategy_file.exists(), f"Strategy file not found: {strategy_file}"

        print("=" * 70)
        print("Cross-chain Lagoon vault manual test")
        print("=" * 70)
        print(f"  Mode:           {'SIMULATE (Anvil forks)' if simulate else 'LIVE'}")
        print(f"  Network:        {network_config['network']}")
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
            network_config=network_config,
            private_key=private_key,
            json_rpc_arbitrum=json_rpc_arbitrum,
            json_rpc_base=json_rpc_base,
            strategy_file=strategy_file,
            usdc_amount=usdc_amount,
            bridge_amount=bridge_amount,
            swap_amount=swap_amount,
            reverse_bridge_amount=reverse_bridge_amount,
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
    network_config: dict,
    private_key: str,
    json_rpc_arbitrum: str,
    json_rpc_base: str,
    strategy_file: Path,
    usdc_amount: Decimal,
    bridge_amount: str,
    swap_amount: str,
    reverse_bridge_amount: str,
    trading_strategy_api_key: str,
    attestation_timeout: float,
):
    """Run the full test lifecycle (extracted to avoid deep nesting)."""
    rpc_env_key_arbitrum, rpc_env_key_base = network_config["rpc_env_keys"]
    source_chain_id, dest_chain_id = network_config["chain_ids"]
    source_chain_name, dest_chain_name = network_config["chain_names"]
    _, dest_chain_slug = network_config["chain_slugs"]
    is_testnet = network_config["is_testnet"]

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
            rpc_env_key_arbitrum: json_rpc_arbitrum,
            rpc_env_key_base: json_rpc_base,
            "VAULT_RECORD_FILE": vault_record_file,
            "FUND_NAME": "Test Crosschain Vault",
            "FUND_SYMBOL": "TCV",
            "ANY_ASSET": "true",
            "PERFORMANCE_FEE": "0",
            "MANAGEMENT_FEE": "0",
            "UNIT_TESTING": "true",
            "LOG_LEVEL": "info",
        }

        run_cli(["lagoon-deploy-vault"], deploy_env)

        # Read deployment record
        deployment_json = vault_record_file.replace(".txt", ".json")
        with open(deployment_json) as f:
            deployment_data = json.load(f)

        arb_dep = deployment_data["deployments"][network_config["chain_slugs"][0]]
        base_dep = deployment_data["deployments"][dest_chain_slug]
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
            rpc_env_key_arbitrum: json_rpc_arbitrum,
            rpc_env_key_base: json_rpc_base,
            "ASSET_MANAGEMENT_MODE": "lagoon",
            "VAULT_ADDRESS": vault_address,
            "VAULT_ADAPTER_ADDRESS": arb_module,
            "UNIT_TESTING": "true",
            "LOG_LEVEL": "info",
            "CACHE_PATH": cache_path,
            "SATELLITE_MODULES": json.dumps({dest_chain_slug: base_module}),
            "MIN_GAS_BALANCE": "0.0",
            "BRIDGE_AMOUNT": bridge_amount,
            "SWAP_AMOUNT": swap_amount,
            "REVERSE_BRIDGE_AMOUNT": reverse_bridge_amount,
        }
        if trading_strategy_api_key:
            base_env["TRADING_STRATEGY_API_KEY"] = trading_strategy_api_key

        run_cli(["init"], {**base_env, "NAME": "Test Crosschain Vault"})

        assert Path(state_file).exists(), "State file was not created"
        print(f"  State file: {state_file}")
        # No wait needed after init — it does not make on-chain transactions

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

        _check_shares(arb_web3, vault_address, deployer.address, "After fund_lagoon_vault")

        # ===================================================================
        # Step 4: Settle vault (process deposit, update NAV)
        # ===================================================================
        print("\n=== Step 4: Settle vault ===")

        settle_env = {k: v for k, v in base_env.items() if k != "NAME"}
        settle_env["SYNC_INTEREST"] = "false"

        # Live RPCs can be slow to propagate new contract state.
        # Retry the settle step with increasing delays.
        # The block cache is cleared in run_cli() before each call.
        import time as _time

        for attempt in range(1, 6):
            if not simulate:
                delay = 30 * attempt
                logger.info("Waiting %ds for live RPC state propagation (attempt %d/5)", delay, attempt)
                _time.sleep(delay)

            try:
                run_cli(["lagoon-settle"], settle_env)
                break
            except Exception as e:
                if attempt == 5:
                    raise
                logger.warning("Settle attempt %d failed: %s. Retrying...", attempt, e)

        print("  Vault settled")
        _check_shares(arb_web3, vault_address, deployer.address, "After Step 4 settle")

        # ===================================================================
        # Step 5: Cycle 1 — Bridge USDC from source chain to destination chain
        # ===================================================================
        print(f"\n=== Step 5: Cycle 1 — Bridge USDC to {dest_chain_name} ===")

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
        bridge_amount_raw = arb_usdc.convert_to_raw(bridge_trade.planned_reserve)
        print(f"  Bridge trade: {bridge_trade.get_status()}")
        print(f"  Burn TX: {burn_tx_hash}")
        print(f"  Amount: {bridge_trade.planned_reserve} USDC")

        # ===================================================================
        # Step 5b: CCTP attestation + receive on destination chain
        # ===================================================================
        print(f"\n=== Step 5b: CCTP attestation + receive on {dest_chain_name} ===")

        if simulate:
            spoof_cctp_attestation(
                dest_web3=base_web3,
                source_chain_id=source_chain_id,
                dest_chain_id=dest_chain_id,
                is_testnet=is_testnet,
                mint_recipient=safe_address,
                amount_raw=bridge_amount_raw,
                deployer=deployer,
                test_attester=test_attesters[dest_chain_id],
            )
        else:
            complete_cctp_bridge(
                dest_web3=base_web3,
                source_chain_id=source_chain_id,
                dest_chain_id=dest_chain_id,
                burn_tx_hash=burn_tx_hash,
                mint_recipient=safe_address,
                amount_raw=bridge_amount_raw,
                deployer=deployer,
                attestation_timeout=attestation_timeout,
            )

        # Verify USDC arrived on the destination chain
        base_usdc = fetch_erc20_details(base_web3, USDC_NATIVE_TOKEN[dest_chain_id])
        base_safe_usdc = wait_for_token_balance(
            base_usdc, safe_address, simulate=simulate, label=f"{dest_chain_name} Safe USDC",
        )
        print(f"  {dest_chain_name} Safe USDC after bridge: {base_safe_usdc}")

        if swap_amount != "0":
            # ===================================================================
            # Step 6: Cycle 2 — Swap USDC -> WETH on destination chain
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
            # Step 7: Cycle 3 — Sell WETH -> USDC on destination chain
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
        else:
            print("\n=== Step 6-7: Skipped (SWAP_AMOUNT=0) ===")

        # ===================================================================
        # Step 8: Cycle 4 — Bridge USDC back to the source chain
        # When SWAP_AMOUNT=0, strategy skips WETH and goes straight to
        # reverse bridge in the next cycle.
        #
        # Override REVERSE_BRIDGE_AMOUNT with the actual Base USDC balance
        # so that all USDC is bridged back in one go. This is needed because
        # swap fees cause a slight mismatch between bridged and available amounts.
        # ===================================================================
        print(f"\n=== Step 8: Cycle 4 — Bridge USDC back to {source_chain_name} ===")

        actual_base_usdc = base_usdc.fetch_balance_of(safe_address)
        bridge_back_env = {**start_env, "REVERSE_BRIDGE_AMOUNT": str(actual_base_usdc)}
        print(f"  Base Safe USDC available: {actual_base_usdc}")

        run_cli(["start"], bridge_back_env)

        state = load_state(state_file)
        reverse_bridge_positions = [
            pos for pos in state.portfolio.open_positions.values()
            if pos.pair.is_cctp_bridge() and pos.pair.quote.chain_id == dest_chain_id
        ]
        assert len(reverse_bridge_positions) == 1, \
            f"Expected 1 reverse bridge position after cycle 4, got {len(reverse_bridge_positions)}"

        reverse_trade = list(reverse_bridge_positions[0].trades.values())[0]
        assert reverse_trade.get_status() == TradeStatus.success, \
            f"Reverse bridge trade status: {reverse_trade.get_status()}"

        reverse_burn_tx = reverse_trade.blockchain_transactions[-1].tx_hash
        reverse_amount_raw = arb_usdc.convert_to_raw(reverse_trade.planned_reserve)
        print(f"  Reverse bridge trade: {reverse_trade.get_status()}")
        print(f"  Burn TX: {reverse_burn_tx}")
        print(f"  Amount: {reverse_trade.planned_reserve} USDC")

        # ===================================================================
        # Step 8b: CCTP attestation + receive on source chain
        # ===================================================================
        print(f"\n=== Step 8b: CCTP attestation + receive on {source_chain_name} ===")

        if simulate:
            spoof_cctp_attestation(
                dest_web3=arb_web3,
                source_chain_id=dest_chain_id,
                dest_chain_id=source_chain_id,
                is_testnet=is_testnet,
                mint_recipient=safe_address,
                amount_raw=reverse_amount_raw,
                deployer=deployer,
                test_attester=test_attesters[source_chain_id],
                nonce=999_999_001,
            )
        else:
            complete_cctp_bridge(
                dest_web3=arb_web3,
                source_chain_id=dest_chain_id,
                dest_chain_id=source_chain_id,
                burn_tx_hash=reverse_burn_tx,
                mint_recipient=safe_address,
                amount_raw=reverse_amount_raw,
                deployer=deployer,
                attestation_timeout=attestation_timeout,
            )

        # Verify USDC returned to the source chain
        arb_safe_usdc_after = wait_for_token_balance(
            arb_usdc, safe_address, simulate=simulate, label=f"{source_chain_name} Safe USDC",
        )
        print(f"  {source_chain_name} Safe USDC after reverse bridge: {arb_safe_usdc_after}")

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
        _check_shares(arb_web3, vault_address, deployer.address, "After Step 10 settle")

        # Check balances
        final_arb_safe_usdc = arb_usdc.fetch_balance_of(safe_address)
        final_base_safe_usdc = base_usdc.fetch_balance_of(safe_address)

        print()
        print("=" * 70)
        print("Final status")
        print("=" * 70)
        print(f"  {source_chain_name} Safe USDC:  {final_arb_safe_usdc}")
        print(f"  {dest_chain_name} Safe USDC: {final_base_safe_usdc}")
        print(f"  Portfolio equity: {final_equity}")
        print(f"  Open positions:  {len(state.portfolio.open_positions)}")
        print(f"  Closed positions: {len(state.portfolio.closed_positions)}")

        # Verify the round trip was successful
        assert final_arb_safe_usdc > 0, f"No USDC returned to {source_chain_name} Safe"

        # Total USDC across all chains should approximately equal the deposited amount
        # (small difference expected from Uniswap swap fees/slippage on the WETH round-trip)
        total_usdc = final_arb_safe_usdc + final_base_safe_usdc
        print(f"  Total USDC across chains: {total_usdc}")
        tolerance = Decimal("0.05")
        assert abs(total_usdc - usdc_amount) < tolerance, \
            f"Total USDC across chains ({total_usdc}) != deposited amount ({usdc_amount}), diff={total_usdc - usdc_amount}"

        # Portfolio equity should account for all cross-chain positions
        state = load_state(state_file)
        final_equity = state.portfolio.get_total_equity()
        print(f"  Portfolio equity (post-settle): {final_equity}")
        assert abs(final_equity - float(usdc_amount)) < float(tolerance), \
            f"Portfolio equity ({final_equity}) != deposited amount ({usdc_amount}), diff={final_equity - float(usdc_amount)}"

        # Display all positions and trades
        all_positions = list(state.portfolio.open_positions.values()) + \
            list(state.portfolio.closed_positions.values())

        print()
        print("=" * 70)
        print("Positions and trades")
        print("=" * 70)

        for pos in all_positions:
            status = "OPEN" if pos.is_open() else "CLOSED"
            print(f"\n  [{status}] Position #{pos.position_id}: {pos.pair.get_human_description()}")
            print(f"    Kind:     {pos.pair.kind.value}")
            print(f"    Value:    ${pos.get_value():.4f}")
            print(f"    Equity:   ${pos.get_equity():.4f}")
            print(f"    Quantity: {pos.get_quantity()}")

            for tid, trade in pos.trades.items():
                print(f"    Trade #{trade.trade_id}: {trade.get_action_verb()} "
                      f"{trade.get_position_quantity()} {pos.pair.base.token_symbol} "
                      f"at ${trade.executed_price or trade.planned_price:.4f} — "
                      f"{trade.get_status().name}")
                if trade.blockchain_transactions:
                    last_tx = trade.blockchain_transactions[-1]
                    print(f"      TX: {last_tx.tx_hash}")

        # ===================================================================
        # Step 11: Redeem vault shares and verify USDC returns to deployer
        #
        # All USDC should have been bridged back to Arb in Step 8.
        # ===================================================================
        print("\n=== Step 11: Redeem vault shares ===")

        deployer.sync_nonce(arb_web3)
        deployer_usdc_before = arb_usdc.fetch_balance_of(deployer.address)
        print(f"  Deployer USDC before redeem: {deployer_usdc_before}")

        # Phase 1: Approve shares and request redemption
        vault = redeem_vault_shares(
            web3=arb_web3,
            vault_address=vault_address,
            redeemer=deployer.address,
            hot_wallet=deployer,
        )
        print("  Redemption requested")

        # Phase 2: Settle via CLI (processes the redemption)
        run_cli(["lagoon-settle"], settle_env)
        print("  Vault settled for redemption")

        # Phase 3: Claim redeemed USDC
        deployer.sync_nonce(arb_web3)
        finalise_fn = vault.finalise_redeem(deployer.address)
        tx_hash = deployer.transact_and_broadcast_with_contract(finalise_fn)
        assert_transaction_success_with_explanation(arb_web3, tx_hash)

        deployer_usdc_after = arb_usdc.fetch_balance_of(deployer.address)
        redeemed_usdc = deployer_usdc_after - deployer_usdc_before
        print(f"  Deployer USDC after redeem: {deployer_usdc_after}")
        print(f"  USDC redeemed: {redeemed_usdc}")

        # Verify deployer got back approximately what was deposited
        # (minus trade losses and decimal conversion errors)
        loss = usdc_amount - redeemed_usdc
        print(f"  Loss from round-trip: {loss} USDC")
        assert redeemed_usdc > 0, "Deployer received no USDC from redemption"
        assert abs(loss) < tolerance, \
            f"Redeemed USDC ({redeemed_usdc}) differs from deposited ({usdc_amount}) by {loss}, exceeds tolerance {tolerance}"

        # Verify shares are fully redeemed
        remaining_shares = vault.share_token.fetch_raw_balance_of(deployer.address)
        assert remaining_shares == 0, f"Deployer still holds {remaining_shares} raw shares after full redemption"

        print()
        print("All ok!")


if __name__ == "__main__":
    main()
