"""Manual test trade for Ostium V1.5 async vault using trade-executor internals.

Exercises the full deposit/redeem lifecycle through the trade-executor's
PositionManager, EthereumExecution, and vault routing — the same code path
used by the ``start`` and ``perform-test-trade`` CLI commands.

Self-contained: builds the trading universe from the on-chain vault contract
without requiring a strategy module or TradingStrategy API key.

Simulation mode
---------------

Set ``SIMULATE=true`` to run on an Anvil mainnet fork with a test wallet.
The script will:
1. Fork Arbitrum at a recent block
2. Fund a test wallet with ETH + USDC
3. Deposit USDC into Ostium vault (requestDeposit)
4. Force settlement on Anvil
5. Claim deposit (settlement retry)
6. Redeem vault shares (requestWithdraw)
7. Force settlement(s) for withdrawal
8. Claim withdrawal (settlement retry)
9. Verify USDC returned and portfolio accounting

Live mode
---------

Without ``SIMULATE=true``, the script runs against real Arbitrum mainnet.
Set ``ACTION`` to control what happens:

- ``status``: Show vault state and owner balances (default)
- ``deposit``: Deposit USDC into the vault (requires confirmation).
  Saves state to ``STATE_FILE`` so ``claim`` can resume later.
- ``claim``: Load state from ``STATE_FILE`` and run settlement retry
  to claim any resolved deposits/withdrawals.
- ``redeem``: Redeem vault shares (requires confirmation)

Environment variables
---------------------

``SIMULATE``
    Set to ``true`` to use an Anvil fork with auto-funded test wallet.

``ACTION``
    One of: ``status``, ``deposit``, ``claim``, ``redeem``, ``simulate_all``.
    Default: ``simulate_all`` if SIMULATE, else ``status``.

``JSON_RPC_ARBITRUM``
    Arbitrum RPC URL.

``PRIVATE_KEY``
    Private key for signing. Not needed in simulation mode.
    Can be a name like ``GMX_PRIVATE_KEY`` which will be resolved
    from the environment.

``VAULT_ADDRESS``
    Ostium vault address (default: OLP vault).

``AMOUNT``
    USDC amount for deposit, or OLP share amount for redeem.
    Default: ``5``.

``STATE_FILE``
    Path to save/load trade-executor state for deposit→claim lifecycle.
    Default: ``/tmp/ostium-v15-test-trade-state.json``.

Usage
-----

Simulated (Anvil fork):

.. code-block:: shell

    source .local-test.env && SIMULATE=true \\
    JSON_RPC_ARBITRUM="https://arb1.arbitrum.io/rpc" \\
    poetry run python scripts/manual-ostium-v15-test-trade.py

Live deposit:

.. code-block:: shell

    source .local-test.env && ACTION=deposit AMOUNT=5 \\
    PRIVATE_KEY=GMX_PRIVATE_KEY \\
    poetry run python scripts/manual-ostium-v15-test-trade.py

Claim after settlement (~24h later):

.. code-block:: shell

    source .local-test.env && ACTION=claim \\
    PRIVATE_KEY=GMX_PRIVATE_KEY \\
    poetry run python scripts/manual-ostium-v15-test-trade.py
"""

import logging
import os
import sys
from decimal import Decimal
from pathlib import Path

from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.classification import create_vault_instance_autodetect
from eth_defi.erc_4626.vault_protocol.gains.vault import OstiumVault, OstiumVersion
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fork_network_anvil, AnvilLaunch
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import fetch_erc20_details, USDC_NATIVE_TOKEN, USDC_WHALE
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.ethereum.vault.settlement_retry import check_and_resolve_vault_settlements
from tradeexecutor.ethereum.vault.vault_utils import translate_vault_to_trading_pair
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.reverse_universe import create_universe_from_trading_pair_identifiers
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse, Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)


# ── Default vault address (Ostium OLP on Arbitrum) ───────────────────
OSTIUM_VAULT_ADDRESS = "0x20d419a8e12c45f88fda7c5760bb6923cee27f98"


def confirm(prompt: str) -> bool:
    """Ask for y/n confirmation before sending a transaction."""
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer == "y"


def resolve_private_key() -> str:
    """Resolve PRIVATE_KEY from env, supporting indirection (e.g. PRIVATE_KEY=GMX_PRIVATE_KEY)."""
    raw = os.environ.get("PRIVATE_KEY", "")
    if not raw:
        print("ERROR: Set PRIVATE_KEY environment variable")
        sys.exit(1)
    # If the value looks like an env var name (no 0x prefix), resolve it
    if not raw.startswith("0x"):
        resolved = os.environ.get(raw, "")
        if resolved:
            logger.info("Resolved PRIVATE_KEY via %s", raw)
            return resolved
        print(f"ERROR: PRIVATE_KEY={raw} but {raw} is not set in environment")
        sys.exit(1)
    return raw


def build_universe(vault, arb_usdc):
    """Build a self-contained TradingStrategyUniverse from an on-chain vault.

    No API key or strategy module needed — reads vault metadata directly
    from the contract and constructs a minimal universe with one pair.
    """
    # Translate on-chain vault to a TradingPairIdentifier
    vault_pair = translate_vault_to_trading_pair(vault)

    # Build exchange universe with a single synthetic exchange entry
    exchange_universe = ExchangeUniverse(
        exchanges={
            1: Exchange(
                chain_id=ChainId(42161),
                chain_slug="arbitrum",
                exchange_id=1,
                exchange_slug="ostium",
                address="0x0000000000000000000000000000000000000000",
                exchange_type=ExchangeType.erc_4626_vault,
                pair_count=1,
            ),
        }
    )

    pair_universe = create_universe_from_trading_pair_identifiers(
        [vault_pair],
        exchange_universe,
    )

    universe = Universe(
        time_bucket=TimeBucket.not_applicable,
        chains={ChainId.arbitrum},
        pairs=pair_universe,
        exchange_universe=exchange_universe,
    )

    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[arb_usdc],
    ), vault_pair


def print_vault_state(vault, web3, owner_address=None):
    """Print vault state summary — mirrors the eth_defi script output."""
    block = web3.eth.block_number
    contract = vault.vault_contract

    print("=" * 70)
    print("OSTIUM V1.5 VAULT — TRADE-EXECUTOR TEST TRADE")
    print("=" * 70)

    print(f"\nChain:          Arbitrum (chain ID: {web3.eth.chain_id})")
    print(f"Block:          {block:,}")
    print(f"Vault:          {vault.name}")
    print(f"Address:        {vault.address}")
    print(f"Share token:    {vault.share_token.symbol}")

    total_assets = vault.fetch_total_assets(block)
    share_price = vault.fetch_share_price(block)
    print(f"\nTVL:             {total_assets} {vault.denomination_token.symbol}")
    print(f"Share price:     {share_price} {vault.denomination_token.symbol}/{vault.share_token.symbol}")

    last_settlement_id = contract.functions.lastSettlementId().call()
    deposit_target = contract.functions.targetSettlementId(True).call()
    withdraw_target = contract.functions.targetSettlementId(False).call()
    print(f"\nLast settlement: {last_settlement_id}")
    print(f"Deposit target:  {deposit_target}")
    print(f"Withdraw target: {withdraw_target}")

    if owner_address:
        eth_balance = web3.from_wei(web3.eth.get_balance(owner_address), "ether")
        usdc_balance = vault.denomination_token.fetch_balance_of(owner_address)
        share_balance = vault.share_token.fetch_balance_of(owner_address)
        print(f"\nOwner: {owner_address}")
        print(f"  ETH:   {eth_balance}")
        print(f"  USDC:  {usdc_balance}")
        print(f"  Shares: {share_balance}")

    print("=" * 70)
    print()


def do_test_trade(web3, hot_wallet, vault, arb_usdc, amount, state_file_path, is_live=False):
    """Execute a test deposit through the trade-executor pipeline.

    Uses PositionManager → EthereumExecution → VaultRouting — the exact same
    code path that the ``start`` CLI command uses for live trading.

    On live mode, saves state to ``state_file_path`` so the ``claim`` action
    can load it later and run settlement retry after off-chain settlement.
    """
    # Build self-contained universe from on-chain vault data
    strategy_universe, vault_pair = build_universe(vault, arb_usdc)

    # Create execution model — mainnet_fork=False for live Arbitrum so the
    # confirmation helper doesn't call Anvil-only evm_mine on slow receipts.
    tx_builder = HotWalletTransactionBuilder(web3, hot_wallet)
    execution_model = EthereumExecution(
        tx_builder,
        mainnet_fork=not is_live,
        confirmation_block_count=0,
    )
    sync_model = HotWalletSyncModel(web3, hot_wallet)

    # Create routing and pricing models
    routing_model = execution_model.create_default_routing_model(strategy_universe)
    pair_configurator = EthereumPairConfigurator(
        web3, strategy_universe, execution_model=execution_model,
    )
    pricing_model = GenericPricing(pair_configurator)

    # Refuse to overwrite a state file that has unresolved pending trades —
    # a second deposit would lose the first trade's ticket metadata, making
    # the first deposit unclaimable via the script's claim action.
    if is_live and state_file_path.exists():
        existing = State.read_json_file(state_file_path)
        pending = [
            t for p in existing.portfolio.open_positions.values()
            for t in p.trades.values()
            if t.get_status() == TradeStatus.vault_settlement_pending
        ]
        if pending:
            print(f"ERROR: State file {state_file_path} has {len(pending)} unresolved pending trade(s).")
            print(f"Run ACTION=claim first, or delete the file to discard the old state.")
            for t in pending:
                print(f"  Trade #{t.trade_id}: {t.other_data.get('vault_direction', '?')}")
            sys.exit(1)

    # Initialise fresh state and sync on-chain reserves
    state = State()
    sync_model.sync_initial(state, reserve_asset=arb_usdc, reserve_token_price=1.0)
    sync_model.sync_treasury(native_datetime_utc_now(), state, supported_reserves=[arb_usdc])

    reserve = state.portfolio.get_default_reserve_position()
    print(f"\nReserve: {float(reserve.quantity):.2f} USDC")
    starting_equity = state.portfolio.calculate_total_equity()
    print(f"Starting equity: {starting_equity:.2f} USDC")

    # Create routing state for trade execution
    routing_state_details = execution_model.get_routing_state_details()
    routing_state = routing_model.create_routing_state(strategy_universe, routing_state_details)
    execution_model.initialize()

    # ── DEPOSIT ──────────────────────────────────────────────────
    print(f"\n--- Depositing {amount} USDC into Ostium vault ---")
    ts = native_datetime_utc_now()
    pm = PositionManager(
        ts, universe=strategy_universe, state=state,
        pricing_model=pricing_model, default_slippage_tolerance=0.10,
    )
    trades = pm.open_spot(vault_pair, value=float(amount))
    execution_model.execute_trades(ts, state, trades, routing_model, routing_state, check_balances=True)

    buy_trade = trades[0]
    status = buy_trade.get_status()
    print(f"Trade status: {status.value}")

    if status == TradeStatus.vault_settlement_pending:
        print(f"Settlement ID stored in trade.other_data")
        pending_value = state.portfolio.get_vault_settlement_pending_value()
        equity = state.portfolio.calculate_total_equity()
        print(f"Pending value: {pending_value:.2f} USDC")
        print(f"Total equity:  {equity:.2f} USDC (should ≈ starting)")

        # Save state so the claim action can load it later
        state.write_json_file(state_file_path)
        print(f"\nState saved to: {state_file_path}")
        print(f"After settlement (~24h), claim with: ACTION=claim")
    elif status == TradeStatus.success:
        print(f"Deposit completed synchronously (unexpected for V1.5)")
    else:
        print(f"ERROR: Unexpected status {status.value}")
        print(f"Revert: {buy_trade.get_revert_reason()}")
        sys.exit(1)

    return state, execution_model


def do_claim(web3, hot_wallet, vault, arb_usdc, state_file_path):
    """Load saved state and run settlement retry to claim resolved deposits/withdrawals.

    After a live deposit enters vault_settlement_pending, settlement happens
    off-chain (~24h). This action loads the state saved by the deposit action,
    runs settlement retry to check on-chain status, and claims if resolved.
    """
    if not state_file_path.exists():
        print(f"ERROR: State file not found: {state_file_path}")
        print(f"Run ACTION=deposit first to create a pending deposit.")
        sys.exit(1)

    # Load state with pending vault trades
    state = State.read_json_file(state_file_path)
    pending_trades = [
        t for p in state.portfolio.open_positions.values()
        for t in p.trades.values()
        if t.get_status() == TradeStatus.vault_settlement_pending
    ]
    if not pending_trades:
        print("No vault_settlement_pending trades found in state.")
        print("All trades may have already been resolved.")
        sys.exit(0)

    print(f"Found {len(pending_trades)} pending vault trade(s)")
    for t in pending_trades:
        print(f"  Trade #{t.trade_id}: {t.other_data.get('vault_direction', '?')} — settlement pending")

    # Create execution model for claiming (live mode, no Anvil mine)
    strategy_universe, _ = build_universe(vault, arb_usdc)
    tx_builder = HotWalletTransactionBuilder(web3, hot_wallet)
    execution_model = EthereumExecution(
        tx_builder,
        mainnet_fork=False,
        confirmation_block_count=0,
    )
    execution_model.initialize()

    # Run settlement retry — checks on-chain status and broadcasts claim txs
    resolved = check_and_resolve_vault_settlements(state=state, execution_model=execution_model)

    if resolved:
        print(f"\nResolved {len(resolved)} trade(s):")
        for t in resolved:
            print(f"  Trade #{t.trade_id}: {t.get_status().value}")
            if t.is_buy():
                print(f"    Received: {t.executed_quantity} shares")
            else:
                print(f"    Received: {t.executed_reserve:.2f} USDC")

        # Save updated state
        state.write_json_file(state_file_path)
        print(f"\nState updated: {state_file_path}")
    else:
        print("\nNo trades resolved — settlement may not have happened yet.")
        print("Try again after the next Ostium settlement epoch (~24h).")


def do_simulate_all(web3, hot_wallet, vault, arb_usdc, amount):
    """Full simulated deposit → settlement → claim → redeem → settlement → claim cycle.

    Uses force_ostium_v15_settlement() to advance the Ostium settlement epoch
    on the Anvil fork, then settlement retry to claim.
    """
    from eth_defi.erc_4626.vault_protocol.gains.testing import force_ostium_v15_settlement

    # Build universe and execution model
    strategy_universe, vault_pair = build_universe(vault, arb_usdc)
    tx_builder = HotWalletTransactionBuilder(web3, hot_wallet)
    execution_model = EthereumExecution(
        tx_builder, mainnet_fork=True, confirmation_block_count=0,
    )
    sync_model = HotWalletSyncModel(web3, hot_wallet)
    routing_model = execution_model.create_default_routing_model(strategy_universe)
    pair_configurator = EthereumPairConfigurator(
        web3, strategy_universe, execution_model=execution_model,
    )
    pricing_model = GenericPricing(pair_configurator)

    # Initialise state
    state = State()
    sync_model.sync_initial(state, reserve_asset=arb_usdc, reserve_token_price=1.0)
    sync_model.sync_treasury(native_datetime_utc_now(), state, supported_reserves=[arb_usdc])

    reserve = state.portfolio.get_default_reserve_position()
    starting_cash = float(reserve.quantity)
    starting_equity = state.portfolio.calculate_total_equity()
    print(f"\nStarting cash:   {starting_cash:.2f} USDC")
    print(f"Starting equity: {starting_equity:.2f} USDC")

    routing_state_details = execution_model.get_routing_state_details()
    routing_state = routing_model.create_routing_state(strategy_universe, routing_state_details)
    execution_model.initialize()

    # ── Step 1: Deposit ──────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"STEP 1: Deposit {amount} USDC → requestDeposit()")
    ts = native_datetime_utc_now()
    pm = PositionManager(
        ts, universe=strategy_universe, state=state,
        pricing_model=pricing_model, default_slippage_tolerance=0.10,
    )
    trades = pm.open_spot(vault_pair, value=float(amount))
    execution_model.execute_trades(ts, state, trades, routing_model, routing_state, check_balances=True)

    buy_trade = trades[0]
    assert buy_trade.get_status() == TradeStatus.vault_settlement_pending, \
        f"Expected vault_settlement_pending, got {buy_trade.get_status()}"
    print(f"  Status: vault_settlement_pending")
    print(f"  Pending value: {state.portfolio.get_vault_settlement_pending_value():.2f} USDC")

    # ── Step 2: Force settlement ─────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"STEP 2: Force settlement on Anvil")
    owner_address = buy_trade.other_data["vault_owner_address"]
    force_ostium_v15_settlement(vault, owner_address)
    print(f"  Settlement forced")

    # ── Step 3: Claim deposit via settlement retry ───────────────────
    print(f"\n{'─' * 70}")
    print(f"STEP 3: Settlement retry → claimDeposit()")
    resolved = check_and_resolve_vault_settlements(state=state, execution_model=execution_model)
    assert len(resolved) == 1
    assert buy_trade.get_status() == TradeStatus.success
    print(f"  Deposit claimed: {buy_trade.executed_quantity} shares")
    print(f"  Pending value: {state.portfolio.get_vault_settlement_pending_value():.2f} USDC (should be 0)")

    # ── Step 4: Redeem ───────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"STEP 4: Redeem all shares → requestWithdraw()")
    ts = native_datetime_utc_now()
    pm = PositionManager(
        ts, universe=strategy_universe, state=state,
        pricing_model=pricing_model, default_slippage_tolerance=0.10,
    )
    trades = pm.close_all()
    execution_model.execute_trades(ts, state, trades, routing_model, routing_state, check_balances=True)

    sell_trade = trades[0]
    assert sell_trade.get_status() == TradeStatus.vault_settlement_pending
    print(f"  Status: vault_settlement_pending")

    # ── Step 5: Force settlement(s) for withdrawal ───────────────────
    print(f"\n{'─' * 70}")
    print(f"STEP 5: Force settlement(s) for withdrawal")
    withdraw_target = vault.vault_contract.functions.targetSettlementId(False).call()
    last_id = vault.vault_contract.functions.lastSettlementId().call()
    settlements_needed = max(withdraw_target - last_id, 1)
    for i in range(settlements_needed):
        force_ostium_v15_settlement(vault, owner_address)
        print(f"  Settlement {i + 1}/{settlements_needed} forced")

    # ── Step 6: Claim withdrawal ─────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"STEP 6: Settlement retry → claimWithdraw()")
    resolved = check_and_resolve_vault_settlements(state=state, execution_model=execution_model)
    assert len(resolved) == 1
    assert sell_trade.get_status() == TradeStatus.success
    print(f"  Withdrawal claimed: {sell_trade.executed_reserve:.2f} USDC")

    # ── Final verification ───────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"VERIFICATION")
    final_equity = state.portfolio.calculate_total_equity()
    final_cash = float(state.portfolio.get_default_reserve_position().quantity)
    print(f"  Starting equity: {starting_equity:.2f} USDC")
    print(f"  Final equity:    {final_equity:.2f} USDC")
    print(f"  Final cash:      {final_cash:.2f} USDC")
    print(f"  Pending value:   {state.portfolio.get_vault_settlement_pending_value():.2f} USDC")
    print(f"  Open positions:  {len(state.portfolio.open_positions)}")
    print(f"  Closed positions: {len(state.portfolio.closed_positions)}")

    equity_diff = abs(final_equity - starting_equity)
    if equity_diff < starting_equity * 0.02:
        print(f"\n  ✓ Equity preserved (diff: {equity_diff:.2f} USDC, <2%)")
    else:
        print(f"\n  ✗ Equity drift too large: {equity_diff:.2f} USDC ({equity_diff / starting_equity * 100:.1f}%)")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"SIMULATION COMPLETE — all steps passed")
    print(f"{'=' * 70}")


# ── Main ─────────────────────────────────────────────────────────────

simulate = os.environ.get("SIMULATE", "").lower() in ("true", "1", "yes")
action = os.environ.get("ACTION", "simulate_all" if simulate else "status").lower()
vault_address = os.environ.get("VAULT_ADDRESS", OSTIUM_VAULT_ADDRESS)
amount = Decimal(os.environ.get("AMOUNT", "5"))
# State file persists vault_settlement_pending trade metadata between deposit and claim
state_file_path = Path(os.environ.get("STATE_FILE", "/tmp/ostium-v15-test-trade-state.json"))
anvil_launch = None

# Reserve asset identifier for Arbitrum USDC
arb_usdc = AssetIdentifier(
    chain_id=42161,
    address=USDC_NATIVE_TOKEN[42161],
    decimals=6,
    token_symbol="USDC",
)

try:
    if simulate:
        # ── Simulation mode: Anvil fork ──────────────────────────────
        print("=" * 70)
        print("OSTIUM V1.5 — TRADE-EXECUTOR TEST TRADE (SIMULATION)")
        print("=" * 70)

        # Fork Arbitrum and fund a test wallet
        usdc_whale = USDC_WHALE[42161]
        anvil_launch = fork_network_anvil(
            os.environ["JSON_RPC_ARBITRUM"],
            fork_block_number=470_000_000,
            unlocked_addresses=[usdc_whale],
        )
        web3 = create_multi_provider_web3(
            anvil_launch.json_rpc_url,
            default_http_timeout=(3, 250.0),
            retries=1,
        )

        hot_wallet = HotWallet.create_for_testing(web3, test_account_n=1, eth_amount=10)
        hot_wallet.sync_nonce(web3)

        # Fund wallet with USDC
        usdc = fetch_erc20_details(web3, USDC_NATIVE_TOKEN[42161])
        fund_amount = max(amount * 10, Decimal(500))
        tx_hash = usdc.contract.functions.transfer(
            hot_wallet.address, int(fund_amount * 10**6),
        ).transact({"from": usdc_whale, "gas": 100_000})
        assert_transaction_success_with_explanation(web3, tx_hash)

        vault = create_vault_instance_autodetect(web3, vault_address)
        assert isinstance(vault, OstiumVault)
        assert vault.version == OstiumVersion.v1_5

        print_vault_state(vault, web3, hot_wallet.address)

        if action == "simulate_all":
            do_simulate_all(web3, hot_wallet, vault, arb_usdc, amount)
        elif action == "deposit":
            do_test_trade(web3, hot_wallet, vault, arb_usdc, amount, state_file_path, is_live=False)
        elif action == "status":
            pass
        else:
            print(f"Unknown ACTION: {action}")
            sys.exit(1)

    else:
        # ── Live mode ────────────────────────────────────────────────
        web3 = create_multi_provider_web3(os.environ["JSON_RPC_ARBITRUM"])
        vault = create_vault_instance_autodetect(web3, vault_address)
        assert isinstance(vault, OstiumVault)
        assert vault.version == OstiumVersion.v1_5

        if action == "status":
            # Show vault state — resolve owner from PRIVATE_KEY if available
            owner = None
            pk = os.environ.get("PRIVATE_KEY", "")
            if pk:
                if not pk.startswith("0x"):
                    pk = os.environ.get(pk, "")
                if pk:
                    owner = HotWallet.from_private_key(pk).address
            print_vault_state(vault, web3, owner)

        elif action in ("deposit", "claim", "redeem"):
            private_key = resolve_private_key()
            hot_wallet = HotWallet.from_private_key(private_key)
            hot_wallet.sync_nonce(web3)

            print_vault_state(vault, web3, hot_wallet.address)

            if action == "deposit":
                if not confirm(f"Deposit {amount} USDC into Ostium vault via trade-executor?"):
                    sys.exit(0)
                # Execute deposit — state is saved to STATE_FILE for later claim
                do_test_trade(
                    web3, hot_wallet, vault, arb_usdc, amount, state_file_path, is_live=True,
                )
                print(f"\nDeposit request submitted. The trade is vault_settlement_pending.")
                print(f"Settlement happens off-chain every ~24h.")
                print(f"After settlement, claim with:")
                print(f"  ACTION=claim PRIVATE_KEY=... poetry run python {sys.argv[0]}")

            elif action == "claim":
                # Load saved state and run settlement retry to claim
                do_claim(web3, hot_wallet, vault, arb_usdc, state_file_path)

            elif action == "redeem":
                print("ERROR: Live redeem not yet supported as standalone — use the start CLI command")
                sys.exit(1)
        else:
            print(f"Unknown ACTION: {action}. Use: status, deposit, claim, redeem, simulate_all")
            sys.exit(1)

finally:
    if anvil_launch is not None:
        print("\nShutting down Anvil fork...")
        anvil_launch.close()
