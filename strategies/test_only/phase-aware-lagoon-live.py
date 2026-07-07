"""Test-only phase-aware Lagoon strategy for live CLI blackbox tests."""

import datetime
import json
import os
from pathlib import Path

from eth_defi.abi import ZERO_ADDRESS_STR
from eth_defi.erc_4626.classification import ERC4626Feature, create_vault_instance
from eth_defi.provider.multi_provider import create_multi_provider_web3

from tradeexecutor.ethereum.vault.testing import PHASE_AWARE_LIVE_E2E_OBSERVATION_KEY, collect_vault_availability
from tradeexecutor.ethereum.vault.vault_utils import translate_vault_to_trading_pair
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.yield_manager import YieldDecisionInput, YieldManager, YieldRuleset, YieldWeightingRule
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.phase_aware import PhaseAwareAlphaModel, queue_vault_pair_ids
from tradeexecutor.strategy.reverse_universe import create_universe_from_trading_pair_identifiers
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.weighting import weight_passthrouh
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.exchange import Exchange, ExchangeType, ExchangeUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


TRADING_STRATEGY_ENGINE_VERSION = "0.5"
trading_strategy_engine_version = TRADING_STRATEGY_ENGINE_VERSION


class Parameters:
    """Static settings for the live CLI test strategy."""

    cycle_duration = CycleDuration.cycle_1s
    chain_id = ChainId.anvil
    routing = TradeRouting.default
    initial_cash = 150
    allocation = 0.95
    cash_change_tolerance_usd = 1.0
    min_trade_threshold = 1.0
    backtest_start = datetime.datetime(2026, 1, 1)
    backtest_end = datetime.datetime(2026, 1, 2)
    time_bucket = TimeBucket.not_applicable


def _normalise_address(address: str) -> str:
    return address.strip().lower()


def _deployment_config() -> dict:
    with Path(os.environ["PHASE_AWARE_LAGOON_DEPLOYMENTS_FILE"]).open("rt") as inp:
        return json.load(inp)


def _target_vault_addresses() -> list[str]:
    return [_normalise_address(vault["address"]) for vault in _deployment_config()["target_vaults"]]


def _queue_vault_address() -> str:
    return _normalise_address(_deployment_config()["queue_vault"])


def _get_pair(strategy_universe: TradingStrategyUniverse, address: str) -> TradingPairIdentifier:
    return strategy_universe.get_pair_by_smart_contract(address)


def _has_open_or_pending_position(state, pair: TradingPairIdentifier) -> bool:
    return any(
        position.pair.internal_id == pair.internal_id
        for position in list(state.portfolio.open_positions.values()) + list(state.portfolio.pending_positions.values())
    )


def _has_target_trade(state, pair: TradingPairIdentifier) -> bool:
    return any(
        trade.pair.internal_id == pair.internal_id
        for trade in state.portfolio.get_all_trades()
    )


def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
) -> IndicatorSet:
    """No indicators - the test drives fixed per-cycle target weights."""
    return IndicatorSet()


def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create a local vault universe from addresses injected by the test."""
    del timestamp, client, execution_context, universe_options

    target_addresses = _target_vault_addresses()
    assert len(target_addresses) == 3, f"Expected three target vault addresses, got {target_addresses}"
    queue_address = _queue_vault_address()

    web3 = create_multi_provider_web3(os.environ["JSON_RPC_ANVIL"])
    vault_pairs = []
    for address in target_addresses + [queue_address]:
        if address in target_addresses:
            features = {ERC4626Feature.lagoon_like, ERC4626Feature.erc_7540_like}
        else:
            features = set()
        vault = create_vault_instance(web3, address, features=features)
        pair = translate_vault_to_trading_pair(vault)
        detected_features = pair.get_vault_features()
        if address in target_addresses:
            assert detected_features is not None, f"Target vault features must be known: {address}"
            assert ERC4626Feature.erc_7540_like in detected_features, f"Target vault must be ERC-7540 async: {address}"
            assert ERC4626Feature.lagoon_like in detected_features, f"Target vault must be Lagoon-like: {address}"
        else:
            assert detected_features == set(), f"Queue vault must be explicitly synchronous: {address}"
        vault_pairs.append(pair)

    exchange = Exchange(
        chain_id=ChainId.anvil,
        chain_slug="anvil",
        exchange_id=1,
        exchange_slug="erc-4626-vault",
        address=ZERO_ADDRESS_STR,
        exchange_type=ExchangeType.erc_4626_vault,
        pair_count=len(vault_pairs),
    )
    exchange_universe = ExchangeUniverse(exchanges={exchange.exchange_id: exchange})
    pair_universe = create_universe_from_trading_pair_identifiers(vault_pairs, exchange_universe)
    universe = Universe(
        time_bucket=TimeBucket.not_applicable,
        chains={ChainId.anvil},
        exchanges={exchange},
        pairs=pair_universe,
        exchange_universe=exchange_universe,
    )
    strategy_universe = TradingStrategyUniverse(data_universe=universe, reserve_assets=[vault_pairs[0].quote])
    strategy_universe.data_universe.pairs.exchange_universe = exchange_universe
    for address in target_addresses:
        pair = strategy_universe.get_pair_by_smart_contract(address)
        pair.other_data["vault_role"] = "phase_aware_target"
    strategy_universe.get_pair_by_smart_contract(queue_address).other_data["vault_role"] = "cash_allocation_queue"
    return strategy_universe


def _make_yield_rules(queue_pair: TradingPairIdentifier) -> YieldRuleset:
    return YieldRuleset(
        position_allocation=Parameters.allocation,
        buffer_pct=0.01,
        cash_change_tolerance_usd=Parameters.cash_change_tolerance_usd,
        weights=[YieldWeightingRule(pair=queue_pair, max_concentration=1.0)],
    )


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Run phase-aware target-vault allocation followed by queue-vault cash management."""
    state = input.state
    strategy_universe = input.strategy_universe
    target_pairs = [_get_pair(strategy_universe, address) for address in _target_vault_addresses()]
    queue_pair = _get_pair(strategy_universe, _queue_vault_address())
    state.other_data.save(
        input.cycle,
        PHASE_AWARE_LIVE_E2E_OBSERVATION_KEY,
        {
            "core_vaults": collect_vault_availability(
                input.pricing_model,
                input.timestamp,
                target_pairs,
            ),
        },
    )

    position_manager = input.get_position_manager()

    yield_rules = _make_yield_rules(queue_pair)
    venue_pair_ids = queue_vault_pair_ids(yield_rules)
    alpha_model = PhaseAwareAlphaModel(input.timestamp, cycle=input.cycle, venue_pair_ids=venue_pair_ids)

    for pair in target_pairs:
        has_position = _has_open_or_pending_position(state, pair)
        has_target_trade = _has_target_trade(state, pair)
        alpha_model.set_signal(pair, 1.0 if not has_position and not has_target_trade else 0.0)

    locked = alpha_model.carry_forward_non_redeemable_positions(position_manager)
    deployable = max(state.portfolio.get_total_equity() * Parameters.allocation - locked, 0.0)

    alpha_model.select_top_signals(count=len(target_pairs))
    alpha_model.assign_weights(method=weight_passthrouh)
    alpha_model.normalise_weights(investable_equity=deployable, max_weight=1.0)
    alpha_model.update_old_weights(
        state.portfolio,
        portfolio_pairs=target_pairs,
        ignore_credit=False,
    )
    alpha_model.calculate_target_positions(position_manager, investable_equity=deployable)
    alpha_model.apply_phase_aware_intent(position_manager)
    trades = alpha_model.generate_rebalance_trades_and_triggers(
        position_manager,
        min_trade_threshold=Parameters.min_trade_threshold,
        individual_rebalance_min_threshold=Parameters.min_trade_threshold,
        sell_rebalance_min_threshold=Parameters.min_trade_threshold,
        execution_context=input.execution_context,
    )
    alpha_model.reconcile_phase_aware_events(position_manager, trades)

    yield_manager = YieldManager(position_manager=position_manager, rules=yield_rules)
    yield_input = YieldDecisionInput(
        execution_mode=input.execution_context.mode,
        cycle=input.cycle,
        timestamp=input.timestamp,
        total_equity=state.portfolio.get_total_equity(),
        directional_trades=trades,
        pending_redemptions=position_manager.get_pending_redemptions(),
    )
    trades += yield_manager.calculate_yield_management_safe(yield_input).trades
    return trades
