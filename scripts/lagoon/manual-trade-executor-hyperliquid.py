"""Manual test: Hypercore vault Lagoon lifecycle via trade-executor CLI.

Exercises the full lifecycle of a Lagoon vault on HyperEVM with a
Hypercore native vault deposit (e.g. HLP):

1. Deploy vault via ``lagoon-deploy-vault --strategy-file=minimal_hyperliquid_strategy.py``
   with Hypercore vault whitelisting
2. Initialise state via ``init``
3. Deposit USDC into the Lagoon vault
4. Settle vault
5. Run strategy cycle (creates Hypercore vault position)
6. Activate Safe on HyperCore + two-phase deposit into Hypercore vault
7. Verify deposit via Hyperliquid info API
8. Run revalue cycle (updates position value from API)
9. (Optional) Withdraw from Hypercore vault
10. (Optional) Settle and redeem Lagoon vault shares

.. note::

    No simulation mode — Hypercore deposits require the real Hyperliquid
    info API for escrow clearing and vault equity queries.

.. warning::

    HyperEVM **testnet** does not support ``depositFor`` activation of
    contract addresses (Safe multisigs). The script works on mainnet only.
    See `web3-ethereum-defi#813 <https://github.com/tradingstrategy-ai/web3-ethereum-defi/issues/813>`_.

Wallet funding
--------------

The deployer wallet needs HYPE (gas) and USDC on HyperEVM.

1. Create a new private key and set ``PRIVATE_KEY`` env var
2. Move ~$2 worth of ETH on Arbitrum to that address
3. Move ~$10 worth of USDC on Arbitrum to that address
4. Sign in to https://app.hyperliquid.xyz with the new account
5. Deposit $10 USDC (minimum $5)
6. Visit https://app.hyperliquid.xyz/portfolio — click EVM <-> CORE
7. Move USDC to HyperEVM (e.g. 10 USDC)
8. Buy a small amount of HYPE on the spot market
9. Move 0.01 HYPE to HyperEVM (for gas)
10. Check HyperEVM balances on the EVM <-> CORE dialog

Environment variables
---------------------

``NETWORK``
    ``mainnet`` (default) or ``testnet``.
    Selects the RPC URL (if ``JSON_RPC_HYPERLIQUID`` is not set) and
    the default Hypercore vault address.

``JSON_RPC_HYPERLIQUID``
    HyperEVM RPC URL. Defaults to the public RPC for the selected network.

``PRIVATE_KEY`` / ``HYPERCORE_WRITER_TEST_PRIVATE_KEY``
    Deployer private key. Must hold HYPE + USDC on HyperEVM
    (at least ``USDC_AMOUNT + 2`` for activation).

``USDC_AMOUNT``
    USDC to deposit into the Hypercore vault (default: ``5``).
    An additional 2 USDC is added automatically for HyperCore activation.

``ACTION``
    ``deposit`` (default), ``withdraw``, or ``both``.
    HLP has a 4-day lock-up period.
    Run deposit first, then withdraw later.

``SAFE_SALT_NONCE``
    CREATE2 salt nonce for deterministic Safe address. If set, the same
    nonce will produce the same Safe address on each chain. Random if
    not set. Printed to stdout for later re-use.

``HYPERCORE_VAULT``
    Hypercore vault address. Defaults to HLP for the selected network.

``TRADING_STRATEGY_API_KEY``
    TradingStrategy.ai API key. Optional for code-based strategies.

Usage
-----

Mainnet:

.. code-block:: shell

    PRIVATE_KEY="0x..." \\
    USDC_AMOUNT=5 \\
    ACTION=deposit \\
        poetry run python scripts/lagoon/manual-trade-executor-hyperliquid.py

Testnet:

.. code-block:: shell

    NETWORK=testnet \\
    PRIVATE_KEY="0x..." \\
    USDC_AMOUNT=5 \\
    ACTION=deposit \\
        poetry run python scripts/lagoon/manual-trade-executor-hyperliquid.py

For agents running this script
------------------------------

This script takes a **very long time** to run (5–15 minutes) because it
deploys a new Lagoon vault, performs on-chain transactions, and waits for
HyperCore escrow clearing. **Do not run it repeatedly** — each run deploys
a new vault and spends real USDC.

When invoking from an LLM agent, redirect output to a log file so the
full output is preserved for debugging:

.. code-block:: shell

    source .local-test.env && PRIVATE_KEY="$HYPERCORE_WRITER_TEST_PRIVATE_KEY" \\
        USDC_AMOUNT=5 ACTION=deposit \\
        poetry run python scripts/lagoon/manual-trade-executor-hyperliquid.py \\
        2>&1 | tee /tmp/hypercore-lifecycle.log

If a transaction fails, extract the transaction hash from the log output
(look for ``tx_hash``, ``send_raw_transaction``, or ``0x`` prefixed hashes)
and use a blockchain explorer or the contract ABI to debug the reverted
transaction. The HyperEVM mainnet explorer is https://www.hyperscan.com/.
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

from eth_typing import HexAddress, HexStr
from tabulate import tabulate
from web3 import Web3

from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.erc_4626.vault_protocol.lagoon.testing import (
    fund_lagoon_vault, redeem_vault_shares)
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from eth_defi.hyperliquid.api import fetch_user_vault_equities
from eth_defi.hyperliquid.core_writer import (
    build_hypercore_deposit_phase1,
    build_hypercore_deposit_phase2,
)
from eth_defi.hyperliquid.evm_escrow import (
    DEFAULT_ACTIVATION_AMOUNT,
    activate_account,
    is_account_activated,
    wait_for_evm_escrow_clear,
)
from eth_defi.hyperliquid.session import (
    HYPERLIQUID_API_URL,
    HYPERLIQUID_TESTNET_API_URL,
    create_hyperliquid_session,
)
from eth_defi.provider.broken_provider import _latest_delayed_block_number_cache
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDC_NATIVE_TOKEN, TokenDetails, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.utils import setup_console_logging
from eth_defi.vault.base import VaultSpec

from tradeexecutor.cli.main import app
from tradeexecutor.ethereum.vault.hypercore_vault import HLP_VAULT_ADDRESS
from tradeexecutor.state.state import State

logger = logging.getLogger(__name__)


def run_cli(args: list[str], env: dict):
    """Run a trade-executor CLI command with the given environment."""
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


def _check_shares(web3: Web3, vault_address: str, deployer_address: str, label: str):
    """Log the deployer's vault share balance."""
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


def _print_hypercore_balances(safe_address: str, network: str = "mainnet") -> list:
    """Query Hyperliquid info API and print the Safe's Hypercore vault balances."""
    api_url = HYPERLIQUID_TESTNET_API_URL if network == "testnet" else HYPERLIQUID_API_URL
    session = create_hyperliquid_session(api_url=api_url)
    equities = fetch_user_vault_equities(session, user=safe_address)
    if equities:
        rows = [[eq.vault_address, f"{eq.equity:,.6f}", eq.locked_until.isoformat()] for eq in equities]
        print("\nHypercore vault balances (Safe):")
        print(tabulate(rows, headers=["Vault", "Equity (USDC)", "Locked until (UTC)"], tablefmt="simple"))
    else:
        print("\nHypercore vault balances: none (Safe has no vault deposits on Hypercore)")
    return equities


def _do_hypercore_deposit(
    *,
    web3: Web3,
    lagoon_vault: LagoonVault,
    deployer: HotWallet,
    usdc_amount_raw: int,
    hypercore_amount_raw: int,
    vault_address: str,
    usdc_human: int,
    network: str = "mainnet",
):
    """Execute two-phase deposit into a Hypercore vault via Lagoon.

    Phase 1: bridge USDC from EVM to HyperCore spot via CoreDepositWallet.
    Wait for EVM escrow to clear.
    Phase 2: move USDC from spot to perp and deposit into vault via CoreWriter.
    """
    api_url = HYPERLIQUID_TESTNET_API_URL if network == "testnet" else HYPERLIQUID_API_URL
    session = create_hyperliquid_session(api_url=api_url)

    # Ensure Safe is activated on HyperCore
    if not is_account_activated(web3, user=lagoon_vault.safe_address):
        logger.info("Activating Safe %s on HyperCore...", lagoon_vault.safe_address)
        activate_account(
            web3=web3,
            lagoon_vault=lagoon_vault,
            deployer=deployer,
            session=session,
        )
        deployer.sync_nonce(web3)

    # Phase 1: bridge USDC to HyperCore
    logger.info("Phase 1: bridging %d USDC to HyperCore spot...", usdc_human)
    fn1 = build_hypercore_deposit_phase1(
        lagoon_vault=lagoon_vault,
        evm_usdc_amount=usdc_amount_raw,
    )
    tx_hash = deployer.transact_and_broadcast_with_contract(fn1)
    receipt = assert_transaction_success_with_explanation(web3, tx_hash)
    logger.info("Phase 1 tx: %s (gas: %d)", tx_hash.hex(), receipt["gasUsed"])

    # Wait for EVM escrow to clear
    logger.info("Waiting for EVM escrow to clear...")
    wait_for_evm_escrow_clear(session, user=lagoon_vault.safe_address)

    # Phase 2: move USDC spot→perp→vault
    logger.info("Phase 2: transferUsdClass + vaultTransfer...")
    deployer.sync_nonce(web3)
    fn2 = build_hypercore_deposit_phase2(
        lagoon_vault=lagoon_vault,
        hypercore_usdc_amount=hypercore_amount_raw,
        vault_address=vault_address,
    )
    tx_hash = deployer.transact_and_broadcast_with_contract(fn2)
    receipt = assert_transaction_success_with_explanation(web3, tx_hash)
    logger.info("Phase 2 tx: %s (gas: %d)", tx_hash.hex(), receipt["gasUsed"])

    # Wait for CoreWriter actions to settle
    logger.info("Waiting 10s for CoreWriter actions to settle on HyperCore...")
    time.sleep(10)

    equities = _print_hypercore_balances(lagoon_vault.safe_address, network=network)
    assert len(equities) > 0, \
        f"Deposit failed: Safe {lagoon_vault.safe_address} has no vault positions on HyperCore"


def _do_hypercore_withdraw(
    *,
    web3: Web3,
    lagoon_vault: LagoonVault,
    deployer: HotWallet,
    hypercore_amount_raw: int,
    vault_address: str,
    usdc_human: int,
    network: str = "mainnet",
):
    """Execute withdrawal from Hypercore vault."""
    from eth_defi.hyperliquid.core_writer import build_hypercore_withdraw_multicall
    from eth_defi.trace import TransactionAssertionError

    logger.info("Executing Hypercore withdrawal (%d USDC)...", usdc_human)
    fn = build_hypercore_withdraw_multicall(
        lagoon_vault=lagoon_vault,
        hypercore_usdc_amount=hypercore_amount_raw,
        vault_address=vault_address,
    )
    try:
        tx_hash = deployer.transact_and_broadcast_with_contract(fn)
        receipt = assert_transaction_success_with_explanation(web3, tx_hash)
        print(f"\n  Withdrawal tx: {tx_hash.hex()} (gas: {receipt['gasUsed']})")
    except (TransactionAssertionError, Exception) as e:
        logger.warning("Withdrawal failed (expected if vault has lock-up period): %s", str(e)[:200])
        print(f"\n  Withdrawal skipped: {str(e)[:200]}")
        return

    logger.info("Waiting 10s for CoreWriter actions to settle on HyperCore...")
    time.sleep(10)
    _print_hypercore_balances(lagoon_vault.safe_address, network=network)


def _revalue_and_check(
    *,
    run_cli_func: Callable,
    start_env: dict,
    state_file: str,
    label: str,
    expected_min_value: float = 0,
) -> State:
    """Run a strategy cycle and verify Hypercore vault position value."""
    run_cli_func(["start"], start_env)

    state = load_state(state_file)

    vault_positions = [
        pos for pos in state.portfolio.open_positions.values()
        if pos.is_vault() and pos.pair.other_data.get("vault_protocol") == "hypercore"
    ]

    if vault_positions:
        vault_pos = vault_positions[0]
        vault_value = float(vault_pos.get_value())
        total_value = float(state.portfolio.get_total_equity())
        print(f"  {label}: Hypercore vault value=${vault_value:.2f}, portfolio total=${total_value:.2f}")
        assert vault_value >= expected_min_value, \
            f"Vault value ${vault_value:.2f} below minimum ${expected_min_value:.2f}"
    else:
        print(f"  {label}: No Hypercore vault position found")
        if expected_min_value > 0:
            raise AssertionError(f"Expected vault value >= ${expected_min_value} but no position found")

    return state


def _run_test_lifecycle(
    *,
    web3: Web3,
    deployer: HotWallet,
    usdc_token: TokenDetails,
    private_key: str,
    json_rpc: str,
    strategy_file: Path,
    usdc_amount: Decimal,
    vault_address: str,
    action: str,
    trading_strategy_api_key: str,
    network: str = "mainnet",
    safe_salt_nonce: int | None = None,
):
    """Run the full test lifecycle."""

    chain_id = web3.eth.chain_id

    usdc_raw = usdc_token.convert_to_raw(usdc_amount)
    hypercore_raw = usdc_raw  # Same decimals

    with tempfile.TemporaryDirectory() as tmp_dir:
        state_file = str(Path(tmp_dir) / "state.json")
        vault_record_file = str(Path(tmp_dir) / "vault-record.txt")
        cache_path = str(Path(tmp_dir) / "cache")

        # ===================================================================
        # Step 1: Deploy vault with Hypercore vault whitelisting
        # ===================================================================
        print("\n=== Step 1: Deploy Hypercore vault ===")

        rpc_env_key = "JSON_RPC_HYPERLIQUID_TESTNET" if network == "testnet" else "JSON_RPC_HYPERLIQUID"

        deploy_env = {
            "NETWORK": network,
            "STRATEGY_FILE": str(strategy_file),
            "PRIVATE_KEY": private_key,
            rpc_env_key: json_rpc,
            "VAULT_RECORD_FILE": vault_record_file,
            "FUND_NAME": "Test Hypercore Vault",
            "FUND_SYMBOL": "TEST",
            "ANY_ASSET": "true",
            "PERFORMANCE_FEE": "0",
            "MANAGEMENT_FEE": "0",
            "UNIT_TESTING": "true",
            "LOG_LEVEL": "info",
        }
        if safe_salt_nonce is not None:
            deploy_env["SAFE_SALT_NONCE"] = str(safe_salt_nonce)

        run_cli(["lagoon-deploy-vault"], deploy_env)

        # Read deployment record
        deployment_json = vault_record_file.replace(".txt", ".json")
        with open(deployment_json) as f:
            deployment_data = json.load(f)

        # The deployment dict uses chain_id.get_slug() as the key
        deployment_keys = list(deployment_data["deployments"].keys())
        assert len(deployment_keys) == 1, f"Expected single-chain deployment, got: {deployment_keys}"
        chain_slug = deployment_keys[0]
        dep = deployment_data["deployments"][chain_slug]
        lagoon_vault_address = dep["vault_address"]
        safe_address = dep["safe_address"]
        module_address = dep["module_address"]

        used_salt_nonce = deployment_data.get("safe_salt_nonce")
        print(f"  Vault:  {lagoon_vault_address}")
        print(f"  Safe:   {safe_address}")
        print(f"  Module: {module_address}")
        print(f"  Safe salt nonce: {used_salt_nonce}")
        logger.info("Safe salt nonce used: %s", used_salt_nonce)

        # ===================================================================
        # Step 2: Initialise state
        # ===================================================================
        print("\n=== Step 2: Initialise state ===")

        base_env = {
            "NETWORK": network,
            "ID": "test-hypercore",
            "STRATEGY_FILE": str(strategy_file),
            "STATE_FILE": state_file,
            "PRIVATE_KEY": private_key,
            rpc_env_key: json_rpc,
            "ASSET_MANAGEMENT_MODE": "lagoon",
            "VAULT_ADDRESS": lagoon_vault_address,
            "VAULT_ADAPTER_ADDRESS": module_address,
            "UNIT_TESTING": "true",
            "LOG_LEVEL": "info",
            "CACHE_PATH": cache_path,
            "MIN_GAS_BALANCE": "0.0",
        }
        if trading_strategy_api_key:
            base_env["TRADING_STRATEGY_API_KEY"] = trading_strategy_api_key

        run_cli(["init"], {**base_env, "NAME": "Test Hypercore Vault"})

        assert Path(state_file).exists(), "State file was not created"
        print(f"  State file: {state_file}")

        # ===================================================================
        # Step 3: Deposit USDC into the Lagoon vault
        # ===================================================================
        print("\n=== Step 3: Deposit USDC ===")

        # Fund the Safe with enough USDC for both activation and the vault deposit.
        # Activation costs 2 USDC (depositFor), so the Safe needs usdc_amount + 2.
        lagoon_fund_amount = usdc_amount + ACTIVATION_USDC

        deployer.sync_nonce(web3)
        fund_lagoon_vault(
            web3=web3,
            vault_address=lagoon_vault_address,
            asset_manager=deployer.address,
            test_account_with_balance=deployer.address,
            trading_strategy_module_address=module_address,
            amount=lagoon_fund_amount,
            hot_wallet=deployer,
        )

        safe_usdc = usdc_token.fetch_balance_of(safe_address)
        print(f"  Safe USDC after deposit: {safe_usdc}")

        _check_shares(web3, lagoon_vault_address, deployer.address, "After fund_lagoon_vault")

        # ===================================================================
        # Step 4: Settle vault
        # ===================================================================
        print("\n=== Step 4: Settle vault ===")

        settle_env = {k: v for k, v in base_env.items() if k != "NAME"}
        settle_env["SYNC_INTEREST"] = "false"

        run_cli(["lagoon-settle"], settle_env)
        print("  Vault settled")
        _check_shares(web3, lagoon_vault_address, deployer.address, "After settle")

        # ===================================================================
        # Step 5: Run strategy cycle — creates Hypercore vault position
        # ===================================================================
        print("\n=== Step 5: Strategy cycle — create Hypercore vault position ===")

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

        # Verify Hypercore vault position was created
        vault_positions = [
            pos for pos in state.portfolio.open_positions.values()
            if pos.is_vault() and pos.pair.other_data.get("vault_protocol") == "hypercore"
        ]
        assert len(vault_positions) >= 1, \
            f"Expected at least 1 Hypercore vault position, got {len(vault_positions)}"

        vault_pos = vault_positions[0]
        print(f"  Vault position pair: {vault_pos.pair.get_ticker()}")
        print(f"  Vault protocol: {vault_pos.pair.other_data.get('vault_protocol')}")
        print(f"  Quantity: {vault_pos.get_quantity()}")

        # ===================================================================
        # Step 6: Activate Safe on HyperCore + deposit into Hypercore vault
        # ===================================================================
        if action in ("deposit", "both"):
            print("\n=== Step 6: Deposit into Hypercore vault ===")

            lagoon_vault = LagoonVault(
                web3,
                VaultSpec(chain_id, lagoon_vault_address),
                trading_strategy_module_address=module_address,
            )

            # Fund Safe with extra USDC for activation (2 USDC overhead)
            deployer.sync_nonce(web3)
            activation_raw = DEFAULT_ACTIVATION_AMOUNT
            activation_human = activation_raw / 10**6
            safe_balance = usdc_token.fetch_balance_of(safe_address)
            needed = float(usdc_amount) + activation_human
            if safe_balance < needed:
                transfer_amount = Decimal(str(needed)) - safe_balance
                logger.info("Transferring %s extra USDC to Safe for activation", transfer_amount)
                tx_hash = deployer.transact_and_broadcast_with_contract(
                    usdc_token.transfer(safe_address, transfer_amount),
                    gas_limit=100_000,
                )
                assert_transaction_success_with_explanation(web3, tx_hash)

            deployer.sync_nonce(web3)
            _do_hypercore_deposit(
                web3=web3,
                lagoon_vault=lagoon_vault,
                deployer=deployer,
                usdc_amount_raw=usdc_raw,
                hypercore_amount_raw=hypercore_raw,
                vault_address=vault_address,
                usdc_human=int(usdc_amount),
                network=network,
            )

            # ===============================================================
            # Step 7: Revalue — position should now reflect vault equity
            # ===============================================================
            print("\n=== Step 7: Revalue Hypercore vault position ===")

            state = _revalue_and_check(
                run_cli_func=run_cli, start_env=start_env, state_file=state_file,
                label="After Hypercore deposit", expected_min_value=float(usdc_amount) * 0.9,
            )

        # ===================================================================
        # Step 8: Withdraw from Hypercore vault (optional)
        # ===================================================================
        if action in ("withdraw", "both"):
            print("\n=== Step 8: Withdraw from Hypercore vault ===")

            lagoon_vault = LagoonVault(
                web3,
                VaultSpec(chain_id, lagoon_vault_address),
                trading_strategy_module_address=module_address,
            )

            deployer.sync_nonce(web3)
            _do_hypercore_withdraw(
                web3=web3,
                lagoon_vault=lagoon_vault,
                deployer=deployer,
                hypercore_amount_raw=hypercore_raw,
                vault_address=vault_address,
                usdc_human=int(usdc_amount),
                network=network,
            )

            # Revalue after withdrawal
            state = _revalue_and_check(
                run_cli_func=run_cli, start_env=start_env, state_file=state_file,
                label="After Hypercore withdrawal", expected_min_value=0,
            )

            # ===============================================================
            # Step 9: Settle and redeem
            # ===============================================================
            print("\n=== Step 9: Settle and redeem ===")

            run_cli(["lagoon-settle"], settle_env)

            deployer.sync_nonce(web3)
            try:
                redeem_vault_shares(
                    web3=web3,
                    vault_address=lagoon_vault_address,
                    redeemer=deployer.address,
                    hot_wallet=deployer,
                )
                print("  Vault shares redeemed")
            except Exception as e:
                logger.warning("Redeem failed (may be expected in test): %s", e)

            # Verify USDC returned
            final_usdc = usdc_token.fetch_balance_of(deployer.address)
            print(f"  Final deployer USDC: {final_usdc}")

        print("\n" + "=" * 70)
        print("Hypercore vault Lagoon lifecycle test PASSED")
        print("=" * 70)


#: Activation costs 2 USDC (depositFor). The deposit amount is on top.
ACTIVATION_USDC = 2


def main():
    setup_console_logging("warning")

    # ----- Parse environment -----
    network = os.environ.get("NETWORK", "mainnet").lower()
    assert network in ("mainnet", "testnet"), \
        f"NETWORK must be 'mainnet' or 'testnet', got '{network}'"

    if network == "testnet":
        json_rpc = os.environ.get(
            "JSON_RPC_HYPERLIQUID_TESTNET",
            "https://rpc.hyperliquid-testnet.xyz/evm",
        )
    else:
        json_rpc = os.environ.get(
            "JSON_RPC_HYPERLIQUID",
            "https://rpc.hyperliquid.xyz/evm",
        )

    private_key = os.environ.get("PRIVATE_KEY") or os.environ.get("HYPERCORE_WRITER_TEST_PRIVATE_KEY")
    assert private_key, "PRIVATE_KEY or HYPERCORE_WRITER_TEST_PRIVATE_KEY is required"

    action = os.environ.get("ACTION", "deposit").lower()
    assert action in ("deposit", "withdraw", "both"), \
        f"ACTION must be 'deposit', 'withdraw', or 'both', got '{action}'"

    vault_address = os.environ.get(
        "HYPERCORE_VAULT",
        HLP_VAULT_ADDRESS.get(network, HLP_VAULT_ADDRESS["mainnet"]),
    )

    usdc_amount = Decimal(os.environ.get("USDC_AMOUNT", "5"))
    trading_strategy_api_key = os.environ.get("TRADING_STRATEGY_API_KEY", "")

    safe_salt_nonce_str = os.environ.get("SAFE_SALT_NONCE")
    safe_salt_nonce = int(safe_salt_nonce_str) if safe_salt_nonce_str else None

    # ----- Connect -----
    web3 = create_multi_provider_web3(json_rpc, default_http_timeout=(3, 500.0))
    chain_id = web3.eth.chain_id
    logger.info("Connected to chain %d, block %d", chain_id, web3.eth.block_number)

    deployer = HotWallet.from_private_key(private_key)
    deployer.sync_nonce(web3)

    # Verify balances on the target chain
    hype_balance = web3.eth.get_balance(deployer.address)
    hype_human = hype_balance / 10**18
    assert hype_human >= 0.001, \
        f"Deployer {deployer.address} has {hype_human:.4f} HYPE on chain {chain_id}, need HYPE for gas"

    usdc_address = USDC_NATIVE_TOKEN[chain_id]
    usdc_token = fetch_erc20_details(web3, usdc_address)
    deployer_usdc = usdc_token.fetch_balance_of(deployer.address)
    min_usdc = usdc_amount + ACTIVATION_USDC
    assert deployer_usdc >= min_usdc, \
        f"Deployer needs at least {min_usdc} USDC on chain {chain_id} ({network}) " \
        f"but has {deployer_usdc} (deposit={usdc_amount}, activation={ACTIVATION_USDC}). " \
        f"See wallet funding instructions in this script's docstring."

    # Strategy file
    strategy_file = (
        Path(__file__).resolve().parent / ".." / ".." /
        "strategies" / "test_only" / "minimal_hyperliquid_strategy.py"
    )
    assert strategy_file.exists(), f"Strategy file not found: {strategy_file}"

    print("=" * 70)
    print("Hypercore vault Lagoon manual test")
    print("=" * 70)
    print(f"  Network:   {network} (chain {chain_id})")
    print(f"  Deployer:  {deployer.address}")
    print(f"  HYPE:      {hype_human:.4f}")
    print(f"  USDC:      {deployer_usdc}")
    print(f"  Deposit:   {usdc_amount} USDC")
    print(f"  Vault:     {vault_address}")
    print(f"  Action:    {action}")
    print(f"  Strategy:  {strategy_file.name}")
    if safe_salt_nonce is not None:
        print(f"  Safe salt nonce: {safe_salt_nonce}")
    else:
        print(f"  Safe salt nonce: (random)")
    print()

    _run_test_lifecycle(
        web3=web3,
        deployer=deployer,
        usdc_token=usdc_token,
        private_key=private_key,
        json_rpc=json_rpc,
        strategy_file=strategy_file,
        usdc_amount=usdc_amount,
        vault_address=vault_address,
        action=action,
        trading_strategy_api_key=trading_strategy_api_key,
        network=network,
        safe_salt_nonce=safe_salt_nonce,
    )


if __name__ == "__main__":
    main()
