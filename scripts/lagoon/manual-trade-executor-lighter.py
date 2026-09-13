"""Live Ethereum Lagoon/Lighter round-trip through trade-executor commands.

This is a manual mainnet test. It deploys a new Lagoon Safe with a generated
Lighter API key, funds and values it through the Typer CLI, opens and closes a
small ETH perpetual, securely withdraws to the Safe and redeems all shares back
to the deployer. It intentionally rejects ``SIMULATE``: an Anvil fork cannot
model Lighter's sequencer, fills or proof-backed withdrawal claim.

Install the optional SDK using the pinned command in
``deps/web3-ethereum-defi/scripts/lagoon/lagoon-lighter-trade-example.py``.
The API private key is read from the mode-0600 deployment record and is never
put in command arguments, environment variables, state, or reports.
"""

import asyncio
import importlib
import json
import logging
import os
import stat
import time
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

from web3 import Web3

from eth_defi.erc_4626.settlement_events import fetch_vault_settlement_logs
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from eth_defi.lighter.api import LIGHTER_MIN_MAINNET_USDC, wait_for_lighter_collateral
from eth_defi.lighter.constants import LIGHTER_API_URL, LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID, LIGHTER_USDC_ETHEREUM
from eth_defi.lighter.lagoon import claim_usdc_to_lagoon_safe_from_lighter, deposit_usdc_from_lagoon_safe_into_lighter
from eth_defi.lighter.session import create_lighter_session
from eth_defi.lighter.valuation import fetch_lighter_account_by_index, fetch_lighter_total_equity
from eth_defi.provider.broken_provider import _latest_delayed_block_number_cache
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.token import fetch_erc20_details
from eth_defi.utils import setup_console_logging
from eth_defi.vault.base import VaultSpec

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State


logger = logging.getLogger(__name__)
ETH_PERP_MARKET_INDEX = 0
USDC_TOLERANCE = Decimal("0.000010")
# Lighter's public equity is a live observation. Funding accrual and display
# rounding can change it between a Typer valuation tick and this follow-up
# read; raw-token transfers below still use ``USDC_TOLERANCE``.
NAV_SYNC_TOLERANCE = Decimal("0.01")
POLL_SECONDS = 5


def require_env(name: str, *fallback_names: str) -> str:
    """Read a required secret/configuration without echoing its value."""
    for key in (name, *fallback_names):
        value = os.environ.get(key)
        if value:
            return value
    raise RuntimeError(f"{name} is required")


def public_error(error: Exception) -> RuntimeError:
    """Avoid propagating an SDK exception which may include signed material."""
    return RuntimeError(f"Lighter operation failed ({type(error).__name__})")


def run_cli(args: list[str], env: dict[str, str]) -> None:
    """Invoke the real Typer application without exposing its patched env."""
    _latest_delayed_block_number_cache.clear()
    logger.info("Running trade-executor %s", " ".join(args))
    patched = dict(env)
    for key in ("PATH", "HOME", "USER", "TMPDIR", "SHELL"):
        if key not in patched and key in os.environ:
            patched[key] = os.environ[key]
    with mock.patch.dict(os.environ, patched, clear=True):
        try:
            app(args, standalone_mode=False)
        except SystemExit as error:
            # Several one-off commands intentionally use sys.exit(0) after a
            # successful account check. Keep the tutorial in control while
            # preserving all non-zero command failures.
            if error.code not in (None, 0):
                raise


def load_secret_record(path: Path) -> dict[str, Any]:
    """Load a generated private record only when its filesystem protection is sane."""
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError(f"Refusing insecure Lighter operator record: {path}")
    return json.loads(path.read_text())


def assert_secret_not_in_public_artifacts(run_dir: Path, secret_record: Path, secret: str) -> None:
    """Ensure deployment's public reports did not accidentally contain the API key."""
    for path in run_dir.iterdir():
        if path == secret_record or not path.is_file() or path.suffix == ".br":
            continue
        if secret in path.read_text(errors="ignore"):
            raise RuntimeError(f"Lighter API key leaked to public artefact {path.name}")


def signed_eth_position(account: dict[str, Any]) -> Decimal:
    """Return the signed ETH-perpetual position from the public account row."""
    for position in account.get("positions") or ():
        if int(position.get("market_id", -1)) == ETH_PERP_MARKET_INDEX:
            size = Decimal(str(position["position"]))
            return size if int(position.get("sign", 1)) >= 0 else -size
    return Decimal(0)


async def wait_for_position(session, account_index: int, expected: str, timeout: int = 300) -> Decimal:
    deadline = time.monotonic() + timeout
    while True:
        value = signed_eth_position(fetch_lighter_account_by_index(session, account_index))
        if (expected == "long" and value > 0) or (expected == "flat" and value == 0):
            return value
        if value < 0:
            raise RuntimeError("Lighter account unexpectedly has an ETH short")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"ETH position did not become {expected}")
        await asyncio.sleep(POLL_SECONDS)


async def resolve_eth_order(lighter: ModuleType, requested_notional: Decimal | None) -> tuple[int, Decimal, int, Decimal]:
    """Return market integer size, human size, scale and effective notional."""
    api_client = lighter.ApiClient(configuration=lighter.Configuration(host=LIGHTER_API_URL))
    try:
        response = await lighter.OrderApi(api_client).order_book_details(market_id=ETH_PERP_MARKET_INDEX)
        market = response.order_book_details[0]
        price = Decimal(str(market.last_trade_price))
        min_quote = Decimal(str(market.min_quote_amount))
        min_base = Decimal(str(market.min_base_amount))
        size_decimals = int(market.size_decimals)
        supported_decimals = int(market.supported_size_decimals)
        target = max(requested_notional or min_quote + Decimal("1"), min_quote, min_base * price)
        step = Decimal(1).scaleb(-supported_decimals)
        base = (max(min_base, target / price) / step).to_integral_value(rounding=ROUND_CEILING) * step
        return int(base * Decimal(10**size_decimals)), base, size_decimals, base * price
    finally:
        await api_client.close()


async def trade(lighter: ModuleType, account_index: int, api_key_index: int, api_private_key: str, base_amount: int, slippage: Decimal, *, is_ask: bool, reduce_only: bool) -> None:
    """Submit one bounded-slippage order without logging signed SDK objects."""
    try:
        client = lighter.SignerClient(url=LIGHTER_API_URL, account_index=account_index, api_private_keys={api_key_index: api_private_key})
        error = client.check_client()
        if error:
            raise RuntimeError("Lighter API key was rejected")
        _, _, error = await client.create_market_order_limited_slippage(
            market_index=ETH_PERP_MARKET_INDEX,
            client_order_index=int(time.time() * 1_000),
            base_amount=base_amount,
            max_slippage=float(slippage),
            is_ask=is_ask,
            reduce_only=reduce_only,
            api_key_index=api_key_index,
        )
        if error:
            raise RuntimeError("Lighter order was rejected")
    except Exception as error:
        raise public_error(error) from None
    finally:
        if "client" in locals():
            await client.close()


async def secure_withdraw(lighter: ModuleType, account_index: int, api_key_index: int, api_private_key: str, amount: Decimal, timeout: int) -> Decimal:
    """Request secure USDC withdrawal and return its final claimable amount."""
    client = None
    api_client = None
    try:
        client = lighter.SignerClient(url=LIGHTER_API_URL, account_index=account_index, api_private_keys={api_key_index: api_private_key})
        _, response, error = await client.withdraw(
            asset_id=client.ASSET_ID_USDC,
            route_type=client.ROUTE_PERP,
            amount=float(amount),
            api_key_index=api_key_index,
        )
        if error or response is None:
            raise RuntimeError("Lighter secure withdrawal was rejected")
        request_hash = response.tx_hash
        api_client = lighter.ApiClient(configuration=lighter.Configuration(host=LIGHTER_API_URL))
        auth, error = client.create_auth_token_with_expiry(api_key_index=api_key_index)
        if error:
            raise RuntimeError("Lighter withdrawal-history authentication failed")
        deadline = time.monotonic() + timeout
        while True:
            history = await lighter.TransactionApi(api_client).withdraw_history(auth, account_index)
            rows = [row for row in history.withdraws if row.asset_id == client.ASSET_ID_USDC and Decimal(row.amount) > 0]
            matches = [row for row in rows if row.l1_tx_hash == request_hash]
            # Older SDK/API rows do not consistently surface the L2 hash; only
            # accept a unique same-amount request created in this invocation.
            if not matches:
                matches = [row for row in rows if abs(Decimal(row.amount) - amount) <= USDC_TOLERANCE]
            if len(matches) == 1 and matches[0].status.lower() == "claimable":
                return Decimal(matches[0].amount)
            if time.monotonic() >= deadline:
                raise TimeoutError("Lighter withdrawal did not become claimable")
            await asyncio.sleep(POLL_SECONDS)
    except Exception as error:
        raise public_error(error) from None
    finally:
        if client is not None:
            await client.close()
        if api_client is not None:
            await api_client.close()


def assert_checkpoint(web3, state_file: Path, vault: LagoonVault, usdc, account_index: int, expected_position: str, start_block: int) -> None:
    """Assert Safe, Lighter, state, and newly posted NAV agree and are non-negative."""
    session = create_lighter_session()
    try:
        equity = fetch_lighter_total_equity(session, account_index)
        position = signed_eth_position(fetch_lighter_account_by_index(session, account_index))
    finally:
        session.close()
    assert equity.get_total().is_finite() and equity.get_total() >= 0, "Negative/non-finite Lighter equity"
    assert (expected_position == "flat" and position == 0) or (expected_position == "long" and position > 0), f"Unexpected ETH position {position}"
    safe_usdc = usdc.fetch_balance_of(vault.safe_address)
    state = State.read_json_file(state_file)
    assert state.portfolio.get_net_asset_value() >= 0, "Negative portfolio NAV"
    reserve = state.portfolio.get_default_reserve_position()
    assert reserve.quantity >= 0 and abs(reserve.quantity - safe_usdc) <= USDC_TOLERANCE, "State reserve does not match Safe USDC"
    positions = [p for p in state.portfolio.get_open_and_frozen_positions() if p.is_exchange_account() and p.pair.get_exchange_account_protocol() == "lighter"]
    assert len(positions) == 1 and positions[0].get_value() >= 0
    assert abs(Decimal(str(positions[0].get_value())) - equity.get_total()) <= NAV_SYNC_TOLERANCE, "State Lighter value mismatch"
    logs = fetch_vault_settlement_logs(web3=web3, address=vault.address, topic0_list=[Web3.to_hex(Web3.keccak(text="NewTotalAssetsUpdated(uint256)"))], start_block=start_block, end_block=web3.eth.block_number, use_hypersync=False)
    assert logs, "lagoon-settle did not post a new NAV"
    posted = usdc.convert_to_decimals(int.from_bytes(bytes(logs[-1]["data"]), byteorder="big"))
    assert abs(posted - (safe_usdc + equity.get_total())) <= NAV_SYNC_TOLERANCE, "Posted NAV does not equal Safe + Lighter equity"


async def main() -> None:
    # CLI/provider INFO diagnostics include full RPC URLs. Those URLs commonly
    # contain credentials, so this live secret-bearing tutorial never enables
    # them from an operator environment variable.
    setup_console_logging("WARNING")
    if os.environ.get("SIMULATE", "").lower() in ("1", "true", "yes"):
        raise RuntimeError("This tutorial is live Ethereum/Lighter only; SIMULATE is unsupported")
    private_key = require_env("LIGHTER_TEST_PRIVATE_KEY", "PRIVATE_KEY")
    rpc = require_env("JSON_RPC_ETHEREUM")
    lighter = importlib.import_module("lighter")
    web3 = create_multi_provider_web3(rpc, default_http_timeout=(3.0, 180.0))
    assert web3.eth.chain_id == LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID
    deployer = HotWallet.from_private_key(private_key)
    usdc = fetch_erc20_details(web3, LIGHTER_USDC_ETHEREUM)
    run_dir = Path(os.environ.get("LIGHTER_RUN_DIR", str(Path("~/.tradingstrategy/examples").expanduser() / f"lighter-manual-{int(time.time())}")))
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    strategy = Path(__file__).resolve().parents[2] / "strategies/test_only/minimal_lighter_strategy.py"
    record = run_dir / "vault-record.txt"
    state_file = run_dir / "state.json"
    base_amount, _, decimals, notional = await resolve_eth_order(
        lighter,
        Decimal(os.environ["LIGHTER_POSITION_USDC"]) if os.environ.get("LIGHTER_POSITION_USDC") else None,
    )
    target = max(Decimal(os.environ.get("LIGHTER_DEPOSIT_USDC", "0")), notional + Decimal("5"), Decimal("5"))
    assert usdc.fetch_balance_of(deployer.address) >= target, "Deployer lacks USDC for the selected Lighter collateral"
    assert web3.eth.get_balance(deployer.address) > 0, "Deployer lacks ETH for mainnet gas"
    base_deploy = {"STRATEGY_FILE": str(strategy), "PRIVATE_KEY": private_key, "JSON_RPC_ETHEREUM": rpc, "VAULT_RECORD_FILE": str(record), "FUND_NAME": "Lighter manual test", "FUND_SYMBOL": "LTM", "ANY_ASSET": "true", "PERFORMANCE_FEE": "0", "MANAGEMENT_FEE": "0", "UNIT_TESTING": "true", "GENERATE_LIGHTER_API_KEY": "true", "LIGHTER_API_KEY_INDEX": os.environ.get("LIGHTER_API_KEY_INDEX", "4"), "LOG_LEVEL": "warning"}
    run_cli(["lagoon-deploy-vault"], base_deploy)
    operator = load_secret_record(record.with_suffix(".json"))
    deployment = operator["deployments"]["ethereum"]
    setup = deployment["lighter_account_setup"]
    account_index, api_key_index, api_private_key = int(setup["account_index"]), int(setup["api_key_index"]), setup["private_key"]
    assert_secret_not_in_public_artifacts(run_dir, record.with_suffix(".json"), api_private_key)
    common = {"ID": "lighter-manual", "STRATEGY_FILE": str(strategy), "STATE_FILE": str(state_file), "PRIVATE_KEY": private_key, "JSON_RPC_ETHEREUM": rpc, "ASSET_MANAGEMENT_MODE": "lagoon", "VAULT_ADDRESS": deployment["vault_address"], "VAULT_ADAPTER_ADDRESS": deployment["module_address"], "LIGHTER_ACCOUNT_INDEX": str(account_index), "UNIT_TESTING": "true", "CACHE_PATH": str(run_dir / "cache"), "MIN_GAS_BALANCE": "0", "LOG_LEVEL": "warning"}
    run_cli(["init"], {**common, "NAME": "Lighter manual test"})
    run_cli(["correct-accounts"], common)
    # Use the same latest-block construction as the Lagoon CLI.  Generic vault
    # classification leaves this metadata lookup unpinned and can fail against
    # a live multi-provider RPC immediately after a deployment.
    vault = LagoonVault(
        web3,
        VaultSpec(web3.eth.chain_id, deployment["vault_address"]),
        trading_strategy_module_address=deployment["module_address"],
        default_block_identifier="latest",
        require_denomination_token=True,
    )
    session = create_lighter_session()
    try:
        current = fetch_lighter_total_equity(session, account_index).get_total()
    finally:
        session.close()
    assert current.is_finite() and current >= 0, "Negative/non-finite Lighter equity before funding"
    additional = target - current
    assert additional >= LIGHTER_MIN_MAINNET_USDC
    deployer.sync_nonce(web3)
    approval = deployer.transact_and_broadcast_with_contract(usdc.approve(vault.address, additional))
    wait_for_transaction_receipt_robust(web3, approval)
    request = deployer.transact_and_broadcast_with_contract(vault.request_deposit(deployer.address, usdc.convert_to_raw(additional)))
    wait_for_transaction_receipt_robust(web3, request)
    run_cli(["lagoon-settle"], common)
    claimable = vault.vault_contract.functions.maxDeposit(deployer.address).call()
    assert claimable > 0
    deployer.sync_nonce(web3)
    claim = deployer.transact_and_broadcast_with_contract(vault.finalise_deposit(deployer.address, raw_amount=claimable))
    wait_for_transaction_receipt_robust(web3, claim)
    safe_before = usdc.fetch_balance_of(vault.safe_address)
    assert safe_before >= additional
    deposit_usdc_from_lagoon_safe_into_lighter(web3, deployer, vault=vault, usdc=usdc, deposit_usdc=additional)
    session = create_lighter_session()
    try:
        wait_for_lighter_collateral(session, account_index, target, timeout=int(os.environ.get("LIGHTER_DEPOSIT_TIMEOUT", "900")))
    finally:
        session.close()
    before = web3.eth.block_number
    run_cli(["lagoon-settle"], common)
    run_cli(["show-valuation"], common)
    assert_checkpoint(web3, state_file, vault, usdc, account_index, "flat", before)
    slippage = Decimal(os.environ.get("LIGHTER_MAX_SLIPPAGE", "0.02"))
    await trade(lighter, account_index, api_key_index, api_private_key, base_amount, slippage, is_ask=False, reduce_only=False)
    session = create_lighter_session()
    try:
        opened = await wait_for_position(session, account_index, "long")
    finally:
        session.close()
    before = web3.eth.block_number
    run_cli(["lagoon-settle"], common)
    run_cli(["show-valuation"], common)
    assert_checkpoint(web3, state_file, vault, usdc, account_index, "long", before)
    close_raw = int((opened * Decimal(10**decimals)).to_integral_value(rounding=ROUND_CEILING))
    await trade(lighter, account_index, api_key_index, api_private_key, close_raw, slippage, is_ask=True, reduce_only=True)
    session = create_lighter_session()
    try:
        await wait_for_position(session, account_index, "flat")
        available = fetch_lighter_total_equity(session, account_index).available_balance
    finally:
        session.close()
    assert available.is_finite() and available >= 0, "Negative/non-finite Lighter available balance"
    before = web3.eth.block_number
    run_cli(["lagoon-settle"], common)
    run_cli(["show-valuation"], common)
    assert_checkpoint(web3, state_file, vault, usdc, account_index, "flat", before)
    withdraw_amount = available.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
    assert withdraw_amount >= LIGHTER_MIN_MAINNET_USDC
    safe_before_claim = usdc.fetch_balance_of(vault.safe_address)
    claimable_amount = await secure_withdraw(lighter, account_index, api_key_index, api_private_key, withdraw_amount, int(os.environ.get("LIGHTER_WITHDRAW_TIMEOUT", "3600")))
    claim_usdc_to_lagoon_safe_from_lighter(web3, deployer, vault=vault, usdc=usdc, claimable_usdc=claimable_amount)
    safe_after_claim = usdc.fetch_balance_of(vault.safe_address)
    assert usdc.convert_to_raw(safe_after_claim - safe_before_claim) == usdc.convert_to_raw(claimable_amount)
    before = web3.eth.block_number
    run_cli(["lagoon-settle"], common)
    run_cli(["show-valuation"], common)
    assert_checkpoint(web3, state_file, vault, usdc, account_index, "flat", before)
    usdc_before_redeem = usdc.fetch_balance_of(deployer.address)
    run_cli(["lagoon-redeem"], common)
    assert vault.share_token.fetch_raw_balance_of(deployer.address) == 0
    assert usdc.fetch_balance_of(deployer.address) > usdc_before_redeem
    assert usdc.fetch_raw_balance_of(vault.safe_address) <= 2, "Unexpected Safe USDC remainder after all-share redemption"
    api_private_key = ""
    logger.info("Lighter manual round trip completed; public run directory: %s", run_dir)


if __name__ == "__main__":
    asyncio.run(main())
