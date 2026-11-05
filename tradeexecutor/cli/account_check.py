"""Read-only custody checks and refusal rules for manual accounting commands.

These rules deliberately do not run in the daemon's shared accounting gate:
startup must be able to resume the very operations that prohibit a correction.
"""

import copy
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
from tradingstrategy.chain import ChainId

from tradeexecutor.cli.bootstrap import configure_lighter_pool_nav
from tradeexecutor.ethereum.address_sync_model import AddressSyncModel
from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.ethereum.lighter.idle_collateral import refresh_lighter_idle_collateral
from tradeexecutor.exchange_account.lighter import create_executor_lighter_session
from tradeexecutor.ethereum.multichain_balance import filter_onchain_assets
from tradeexecutor.ethereum.web3config import Web3Config
from tradeexecutor.state.identifier import AssetIdentifier, normalise_custody_chain_id
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.strategy.asset import build_expected_asset_map
from tradeexecutor.strategy.sync_model import SyncModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def pending_account_trades(state: State) -> list[TradeExecution]:
    """Find routed operations whose wallet balance cannot prove their outcome."""
    return [
        trade for trade in state.portfolio.get_all_trades()
        if (trade.pair.is_cctp_bridge() or (trade.pair.is_vault() and not trade.pair.is_hyperliquid_vault()))
        # A confirmed reclaim is terminal and restores its funding even though
        # the original investment failed. Historical failures must not keep
        # every future cash check excluded from the report.
        and trade.get_status() not in {TradeStatus.success, TradeStatus.failed, TradeStatus.repaired, TradeStatus.expired}
    ]


def pending_account_assets(state: State, trades: list[TradeExecution]) -> set[AssetIdentifier]:
    """Exclude shared cash as well as the tokens affected by pending requests."""
    assets = {asset for trade in trades for asset in (trade.pair.base, trade.pair.quote, trade.reserve_currency)}
    if trades:
        # A satellite payout returns through a bridge position, and a bridge
        # receive can change the hub reserve. Both cash ledgers may still await
        # booking. Checking them as settled balances could double-credit cash.
        assets.update(reserve.asset for reserve in state.portfolio.reserves.values())
        assets.update(p.pair.base for p in state.portfolio.get_all_positions() if p.pair.is_cctp_bridge())
    return assets


def configure_account_check(sync_model: SyncModel, web3config: Web3Config, deployment_file: Path, state: State, universe: TradingStrategyUniverse | None = None) -> bool:
    """Validate all required RPCs and the supported shared-Safe deployment.

    Returns whether the custody spans multiple EVM chains. No signer or
    execution model is created, so inspection works with public metadata only.

    The first call can validate recorded custody before side effects; a later
    call includes configured assets discovered when constructing the universe.
    """
    if isinstance(sync_model, AddressSyncModel):
        sync_model.web3config = web3config
    expected = build_expected_asset_map(state.portfolio, universe.data_universe.pairs if universe is not None else None)
    chains = {asset.chain_id for asset in filter_onchain_assets(expected)}
    if universe is not None:
        chains.update(asset.chain_id for asset in filter_onchain_assets(universe.reserve_assets))
        for pair in universe.iterate_pairs():
            if pair.is_cctp_bridge() or pair.is_lighter_pool():
                chains.add(normalise_custody_chain_id(pair.base.chain_id))
                chains.add(normalise_custody_chain_id(pair.quote.chain_id))
    if isinstance(sync_model, LagoonVaultSyncModel):
        multichain_metadata = False
        required_chains = set(chains)
        metadata_chains: set[int] = set()
        if deployment_file.exists():
            payload = json.loads(deployment_file.read_text())
            multichain_metadata = bool(payload.get("multichain"))
            # Normalised legacy single-chain reports contain vault/module
            # addresses but no Safe field. The on-chain vault identifies that
            # owner; cross-chain ownership validation is needed for satellites.
            deployments = (payload.get("deployments") or {}) if payload.get("multichain") else {}
            for slug, deployment in deployments.items():
                chain = ChainId.get_by_slug(slug)
                if chain is None:
                    raise RuntimeError(f"Unknown custody chain in deployment metadata: {slug}")
                chains.add(chain.value)
                metadata_chains.add(chain.value)
                # The coordinated deployment uses CREATE2 for the same Safe
                # address on every chain. Different owners require another
                # topology, not a best-effort balance read from the hub owner.
                safe = deployment.get("safe_address")
                if not safe or safe.lower() != sync_model.vault.safe_address.lower():
                    raise RuntimeError(f"Conflicting Safe custody metadata for {slug}")
        configure_lighter_pool_nav(sync_model, deployment_file)
        if len(chains) > 1 and not multichain_metadata:
            raise RuntimeError("Missing multichain deployment metadata; cannot validate satellite Safe custody")
        if len(chains) > 1 and not required_chains.issubset(metadata_chains):
            raise RuntimeError("Incomplete multichain deployment metadata; required custody chain is missing")
    for chain in chains:
        try:
            connection = web3config.get_connection(ChainId(chain))
        except KeyError:
            # A sole explicitly configured Anvil fork is a supported unit-test
            # seam. Live commands must always identify the actual custody chain.
            if web3config.unit_testing and len(web3config.connections) == 1 and ChainId.anvil in web3config.connections:
                connection = web3config.get_default()
            else:
                raise RuntimeError(f"No RPC configured for balance custody chain {chain}") from None
        actual = connection.eth.chain_id
        if actual != chain and not (web3config.unit_testing and actual == 31337):
            raise RuntimeError(f"Balance RPC chain mismatch: expected {chain}, got {actual}")
    return len(chains) > 1


def check_pool_accounts(state: State, sync_model: SyncModel, universe: TradingStrategyUniverse) -> tuple[bool, pd.DataFrame]:
    """Inspect pool holdings without updating saved NAV or writing off dust.

    Reuse the NAV validator on a copy because it updates validity and prices.
    Return a clean flag and report rows with expected/actual quantities and
    custody details. Unavailable or invalid observations return an unchecked
    status and a false flag, so callers cannot certify incomplete balances.
    """
    positions = [p for p in state.portfolio.get_open_and_frozen_positions() if p.pair.is_lighter_pool()]
    required = positions or state.portfolio.lighter_idle_collateral or any(p.is_lighter_pool() for p in universe.iterate_pairs())
    account = getattr(sync_model, "lighter_pool_account", None)
    if not required and account is None:
        return True, pd.DataFrame()
    rows = []
    try:
        if account is None:
            raise RuntimeError("Missing public Lighter pool account metadata")
        # Valuation validates ownership, unexpected holdings and stable API
        # observations. Work on a copy: even its invalidation-on-error must not
        # overwrite the evidence the operator is trying to inspect.
        observed = copy.deepcopy(state)
        session = getattr(sync_model, "lighter_pool_session", None) or create_executor_lighter_session()
        refresh_lighter_idle_collateral(observed, session, *account)
        observation = next(
            item for item in observed.portfolio.lighter_idle_collateral.values()
            if (item.custody_chain_id, item.safe_address.lower(), item.account_index) == (1, account[1].lower(), account[0])
        )
        saved = state.portfolio.lighter_idle_collateral.get(observation.get_identifier())
        clean = saved is not None and saved.valid and abs(saved.amount - observation.amount) <= Decimal("0.000001")
        rows.append({"Position": "Lighter idle collateral", "Chain": 1, "Owner": account[1], "Account": account[0], "Expected": saved.amount if saved else "unobserved", "Actual": observation.amount, "Observed at": observation.observed_at, "Status": "match" if clean else "unclean; reconcile pool account"})
        for position in positions:
            # The shared validator has compared the complete raw share map,
            # including unexpected pools. A passed validation proves equality.
            rows.append({"Position": position.pair.get_ticker(), "Chain": 9998, "Owner": account[1], "Account": account[0], "Expected": position.get_quantity(), "Actual": position.get_quantity(), "Observed at": observation.observed_at, "Status": "match"})
        return clean, pd.DataFrame(rows)
    except Exception as exc:
        # Inspection stays useful when one provider fails, but must never
        # certify an incomplete account. Correction refuses this same result.
        return False, pd.DataFrame([{"Position": "Lighter pool account", "Status": f"unchecked: {exc}"}])


def refuse_unreconciled_investors(sync_model: SyncModel, state: State) -> None:
    """Keep investor settlements out of generic unknown-cause corrections."""
    if not isinstance(sync_model, LagoonVaultSyncModel):
        return
    block = sync_model.get_safe_latest_block()
    reserve = state.portfolio.get_default_reserve_position()
    analyses = sync_model._discover_lagoon_settlements(state=state, reserve_position=reserve, safe_sync_block=block)
    deposits, redemptions = sync_model._fetch_lagoon_queue_amounts(block)
    if analyses or deposits or redemptions:
        raise RuntimeError("Unreconciled investor flows; inspect check-accounts and resolve unexplained custody differences before lagoon-settle, then retry correct-accounts")
