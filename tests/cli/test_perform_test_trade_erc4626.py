"""Focused selection and admission checks for public vault diagnostics."""

import dataclasses
import subprocess
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from eth_defi.erc_4626.deposit_redeem import ERC4626DepositManager
from eth_defi.erc_4626.vault import ERC4626Vault
from eth_defi.vault.deposit_redeem import VaultFlowUnavailable
from pytest import MonkeyPatch
from tradingstrategy.chain import ChainId
from typer.main import get_command

from tradeexecutor.cli.main import app
from tradeexecutor.cli.testtrade import make_test_trade, resolve_test_pair
from tradeexecutor.cli.vault_test_trade import check_vault_test_capacity
from tradeexecutor.ethereum.vault.vault_live_pricing import VaultPricing
from tradeexecutor.state.identifier import TradingPairKind
from tradeexecutor.state.state import State
from tradeexecutor.strategy.sync_model import SyncModel
from tradeexecutor.testing.erc4626_manual import STRATEGY_FILE, load_universe


def test_erc4626_pair_selection_and_cli_refusals(monkeypatch: MonkeyPatch) -> None:
    """Resolve identical addresses by chain and reject conflicting CLI modes before RPC use.

    Pair iteration and execution dependencies are mocked to model identical
    addresses and verify unsupported routes stop before treasury or trade calls.

    1. Supply same-address Arbitrum and Base pairs and select the requested chain.
    2. Refuse absent and ambiguous matches.
    3. Refuse conflicting modes and unsupported close-only routes before treasury sync.
    4. Check the manual tutorial's shell syntax and complete declared matrix.
    """
    # 1. Supply same-address Arbitrum and Base pairs and select the requested chain.
    universe = load_universe()
    original = next(universe.iterate_pairs())
    base = dataclasses.replace(original, base=dataclasses.replace(original.base, chain_id=8453), quote=dataclasses.replace(original.quote, chain_id=8453))
    monkeypatch.setattr(universe, "iterate_pairs", lambda: iter([original, base]))
    assert resolve_test_pair(universe, (ChainId.arbitrum, original.pool_address)) is original
    assert resolve_test_pair(universe, (ChainId.base, original.pool_address.upper())) is base

    # 2. Refuse absent and ambiguous matches.
    with pytest.raises(ValueError, match="found 0"):
        resolve_test_pair(universe, (ChainId.ethereum, original.pool_address))
    monkeypatch.setattr(universe, "iterate_pairs", lambda: iter([original, original]))
    with pytest.raises(ValueError, match="found 2"):
        resolve_test_pair(universe, (ChainId.arbitrum, original.pool_address))

    # 3. Refuse conflicting modes and unsupported close-only routes before treasury sync.
    cli = get_command(app)
    for options in (["--buy-only"], ["--all-vaults"]):
        with pytest.raises(ValueError, match="--close-only requires"):
            cli.main(args=["perform-test-trade", "--strategy-file", str(STRATEGY_FILE), "--close-only", "--pair", f"(arbitrum, {original.pool_address})", *options], standalone_mode=False)

    sync_model = MagicMock(spec=SyncModel)
    execution_model = MagicMock()
    for unsupported in (dataclasses.replace(original, kind=TradingPairKind.exchange_account), base):
        with pytest.raises(ValueError, match="home-chain synchronous ERC-4626"):
            make_test_trade(
                web3=MagicMock(), execution_model=execution_model, pricing_model=MagicMock(),
                sync_model=sync_model, state=State(), universe=load_universe(),
                routing_model=MagicMock(), routing_state=MagicMock(), max_slippage=0.005,
                pair=unsupported, close_only=True, test_trade_checks=True,
                web3config=MagicMock(default_chain_id=ChainId.arbitrum),
            )
    sync_model.sync_treasury.assert_not_called()
    execution_model.execute_trades.assert_not_called()

    # 4. Check the manual tutorial's shell syntax and complete declared matrix.
    script = Path(__file__).resolve().parents[2] / "scripts/lagoon/manual-erc4626-cli.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert {p.get_vault_name() for p in load_universe().iterate_pairs()} == {"generic", "morpho_v1", "morpho_v2", "yearn_v3", "compounder"}


def test_shared_capacity_and_empty_quote_are_refused() -> None:
    """Shared manager capacity refusals reach the command before execution.

    The contract reader is mocked to fix owner limits and quotes without
    changing protocol state; the real base manager and pricing methods are used.

    1. Model a generic vault with a fixed owner and raw unit conversion.
    2. Accept positive deposit capacity, then refuse insufficient capacity in both directions.
    3. Reject a zero-share deposit quote before dividing or submitting transactions.
    """
    # 1. Model a generic vault with a fixed owner and raw unit conversion.
    pair = next(load_universe().iterate_pairs())
    vault = MagicMock(spec=ERC4626Vault)
    vault.erc_7540 = False
    vault.is_whitelisted_deposit.return_value = False
    vault.fetch_deposit_closed_reason.return_value = None
    vault.denomination_token = MagicMock(address=pair.quote.address)
    vault.denomination_token.convert_to_raw.side_effect = lambda v: int(v * Decimal(10 ** pair.quote.decimals))
    vault.denomination_token.fetch_raw_balance_of.return_value = 1_000_000
    vault.share_token = MagicMock()
    vault.share_token.convert_to_raw.side_effect = lambda v: int(v * Decimal(10 ** pair.base.decimals))
    vault.share_token.convert_to_decimals.side_effect = lambda v: Decimal(v) / Decimal(10 ** pair.base.decimals)
    vault.share_token.fetch_raw_balance_of.return_value = 1_000_000
    manager = ERC4626DepositManager(vault)
    vault.get_deposit_manager.return_value = manager
    pricing = VaultPricing(MagicMock(), owner_address_resolver=lambda _: "0x0000000000000000000000000000000000000001")
    pricing.get_vault = MagicMock(return_value=vault)

    # 2. Accept positive deposit capacity, then refuse insufficient capacity in both directions.
    vault.vault_contract.functions.maxDeposit.return_value.call.return_value = 1_000_000
    check_vault_test_capacity(pricing, pair, Decimal(1), closing=False)
    vault.vault_contract.functions.maxDeposit.return_value.call.return_value = 999_999
    vault.vault_contract.functions.maxRedeem.return_value.call.return_value = 999_999
    for closing in (False, True):
        with pytest.raises(VaultFlowUnavailable) as failure:
            check_vault_test_capacity(pricing, pair, Decimal(1), closing=closing)
        assert failure.value.available_raw_amount == 999_999

    # 3. Reject a zero-share deposit quote before dividing or submitting transactions.
    vault.vault_contract.functions.previewDeposit.return_value.call.return_value = 0
    with pytest.raises(VaultFlowUnavailable, match="no shares"):
        pricing.get_buy_price(None, pair, Decimal(1))
