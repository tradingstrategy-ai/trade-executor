"""Test-only always-open Lagoon async settlement strategy for live CLI blackbox tests."""

import datetime
import json
import os
from decimal import Decimal
from pathlib import Path

from eth_defi.abi import ZERO_ADDRESS_STR
from eth_defi.erc_4626.classification import ERC4626Feature, create_vault_instance
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.trace import assert_transaction_success_with_explanation
from web3 import Web3

from tradeexecutor.ethereum.vault.testing import LAGOON_ALWAYS_OPEN_LIVE_E2E_OBSERVATION_KEY
from tradeexecutor.ethereum.vault.vault_utils import translate_vault_to_trading_pair
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.yield_manager import YieldDecisionInput, YieldManager, YieldRuleset, YieldWeightingRule
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.phase_aware import PhaseAwareAlphaModel, queue_vault_pair_ids
from tradeexecutor.strategy.reverse_universe import create_universe_from_trading_pair_identifiers
from tradeexecutor.strategy.strategy_module import StrategyTickHookInput
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


_CONTROLLER_STATE = {
    "web3": None,
    "deployment_config": None,
    "owner_address": None,
    "vault_cache": {},
    "pre_tick_onchain_vaults": {},
}


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
    config = _CONTROLLER_STATE.get("deployment_config")
    if config is not None:
        return config
    with Path(os.environ["LAGOON_ALWAYS_OPEN_DEPLOYMENTS_FILE"]).open("rt") as inp:
        config = json.load(inp)
    _CONTROLLER_STATE["deployment_config"] = config
    return config


def _target_vault_configs() -> list[dict]:
    return _deployment_config()["target_vaults"]


def _target_vault_addresses() -> list[str]:
    return [_normalise_address(vault["address"]) for vault in _target_vault_configs()]


def _queue_vault_address() -> str:
    return _normalise_address(_deployment_config()["queue_vault"])


def _target_vault_scenario() -> list[dict]:
    return _deployment_config()["target_vault_scenario"]


def _target_vault_asset_managers() -> list[str]:
    return [vault["asset_manager"] for vault in _target_vault_configs()]


def _target_vault_trading_strategy_modules() -> list[str]:
    return [vault["trading_strategy_module"] for vault in _target_vault_configs()]


def _controller_web3():
    web3 = _CONTROLLER_STATE.get("web3")
    if web3 is None:
        web3 = create_multi_provider_web3(
            os.environ["JSON_RPC_ANVIL"],
            default_http_timeout=(3, 60.0),
            retries=1,
        )
        _CONTROLLER_STATE["web3"] = web3
    return web3


def _controller_owner_address() -> str:
    owner_address = _CONTROLLER_STATE.get("owner_address")
    if owner_address is None:
        owner_address = HotWallet.from_private_key(os.environ["PRIVATE_KEY"]).address
        _CONTROLLER_STATE["owner_address"] = owner_address
    return owner_address


def _get_pair(strategy_universe: TradingStrategyUniverse, address: str) -> TradingPairIdentifier:
    return strategy_universe.get_pair_by_smart_contract(address)


def _has_target_trade(state: State, pair: TradingPairIdentifier) -> bool:
    return any(
        trade.pair.internal_id == pair.internal_id
        for trade in state.portfolio.get_all_trades()
    )


def _trade_vault_address(trade: TradeExecution) -> str:
    return trade.pair.pool_address.lower()


def _trade_direction(trade: TradeExecution) -> str:
    return "deposit" if trade.is_buy() else "redeem"


def _has_claim_transaction(trade: TradeExecution) -> bool:
    return any(tx.other.get("vault_settlement_action") == "claim" for tx in trade.blockchain_transactions)


def _decimal_str(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _datetime_str(value: datetime.datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _raw_str(value: int) -> str:
    return str(int(value))


def _uint256_arg(value: int) -> str:
    return f"{int(value):064x}"


def _address_arg(address: str) -> str:
    return address.removeprefix("0x").lower().zfill(64)


def _eth_call_uint256(receipt_web3, address: str, signature: str, args: str = "") -> int:
    data = "0x" + Web3.keccak(text=signature)[0:4].hex() + args
    result = receipt_web3.eth.call(
        {
            "to": Web3.to_checksum_address(address),
            "data": data,
        },
        block_identifier="latest",
    )
    if len(result) == 0:
        return 0
    return int.from_bytes(result, byteorder="big")


def _load_lagoon_vault(
    receipt_web3,
    target_vault: str,
    trading_strategy_module: str,
    vault_cache: dict[str, object],
):
    vault = vault_cache.get(target_vault)
    if vault is None:
        vault = create_vault_instance(
            receipt_web3,
            target_vault,
            features={ERC4626Feature.lagoon_like, ERC4626Feature.erc_7540_like},
        )
        vault.trading_strategy_module_address = trading_strategy_module
        vault_cache[target_vault] = vault
    return vault


def _collect_vault_observations(
    receipt_web3,
    target_vaults: list[str],
    owner_address: str,
    include_availability: bool,
) -> dict[str, dict]:
    onchain = {}
    owner_arg = _address_arg(owner_address)
    for target_vault in target_vaults:
        request_args = _uint256_arg(0) + owner_arg
        vault_state = {
            "pending_deposit_raw": _raw_str(_eth_call_uint256(receipt_web3, target_vault, "pendingDepositRequest(uint256,address)", request_args)),
            "pending_redeem_raw": _raw_str(_eth_call_uint256(receipt_web3, target_vault, "pendingRedeemRequest(uint256,address)", request_args)),
            "share_balance_raw": _raw_str(_eth_call_uint256(receipt_web3, target_vault, "balanceOf(address)", owner_arg)),
        }
        if include_availability:
            vault_state.update({
                "max_deposit_raw": _raw_str(_eth_call_uint256(receipt_web3, target_vault, "maxDeposit(address)", owner_arg)),
                "max_redeem_raw": _raw_str(_eth_call_uint256(receipt_web3, target_vault, "maxRedeem(address)", owner_arg)),
            })
        onchain[target_vault] = vault_state
    return onchain


def _collect_queue_vault_observation(
    receipt_web3,
    queue_vault: str,
    owner_address: str,
) -> dict:
    owner_arg = _address_arg(owner_address)
    return {
        "share_balance_raw": _raw_str(_eth_call_uint256(receipt_web3, queue_vault, "balanceOf(address)", owner_arg)),
        "max_deposit_raw": _raw_str(_eth_call_uint256(receipt_web3, queue_vault, "maxDeposit(address)", owner_arg)),
        "max_redeem_raw": _raw_str(_eth_call_uint256(receipt_web3, queue_vault, "maxRedeem(address)", owner_arg)),
        "total_assets_raw": _raw_str(_eth_call_uint256(receipt_web3, queue_vault, "totalAssets()")),
    }


def _trade_evidence(trade: TradeExecution) -> dict:
    claim_transaction = _has_claim_transaction(trade)
    return {
        "trade_id": trade.trade_id,
        "vault": _trade_vault_address(trade),
        "pair_internal_id": trade.pair.internal_id,
        "direction": _trade_direction(trade),
        "opened_at": _datetime_str(trade.opened_at),
        "status": trade.get_status().value,
        "pending": trade.get_status() == TradeStatus.vault_settlement_pending,
        "processed": trade.is_success() and claim_transaction,
        "claim_transaction": claim_transaction,
        "planned_reserve": _decimal_str(trade.planned_reserve),
        "planned_quantity": _decimal_str(trade.planned_quantity),
        "executed_reserve": _decimal_str(trade.executed_reserve),
        "executed_quantity": _decimal_str(trade.executed_quantity),
    }


def _record_observation(
    state: State,
    cycle: int,
    cycle_timestamp: datetime.datetime,
    target_vaults: list[str],
    queue_vault: str,
    receipt_web3,
    owner_address: str,
    pre_tick_onchain_vaults: dict[str, dict],
) -> None:
    target_set = set(target_vaults)
    target_trades = [
        _trade_evidence(trade)
        for trade in state.portfolio.get_all_trades()
        if _trade_vault_address(trade) in target_set
    ]
    queue_trades = [
        {
            "trade_id": trade.trade_id,
            "vault": _trade_vault_address(trade),
            "direction": _trade_direction(trade),
            "opened_at": _datetime_str(trade.opened_at),
            "status": trade.get_status().value,
            "yield_decision": trade.get_yield_decision() is not None,
            "planned_reserve": _decimal_str(trade.planned_reserve),
            "planned_quantity": _decimal_str(trade.planned_quantity),
            "executed_reserve": _decimal_str(trade.executed_reserve),
            "executed_quantity": _decimal_str(trade.executed_quantity),
        }
        for trade in state.portfolio.get_all_trades()
        if _trade_vault_address(trade) == queue_vault
    ]
    observation = {
        "cycle": cycle,
        "timestamp": _datetime_str(cycle_timestamp),
        "target_trades": target_trades,
        "pending_deposits": [trade for trade in target_trades if trade["direction"] == "deposit" and trade["pending"]],
        "processed_deposits": [trade for trade in target_trades if trade["direction"] == "deposit" and trade["processed"]],
        "pending_redemptions": [trade for trade in target_trades if trade["direction"] == "redeem" and trade["pending"]],
        "processed_redemptions": [trade for trade in target_trades if trade["direction"] == "redeem" and trade["processed"]],
        "queue_trades": queue_trades,
        "queue_deposits": [trade for trade in queue_trades if trade["direction"] == "deposit" and trade["status"] == TradeStatus.success.value],
        "queue_redemptions": [trade for trade in queue_trades if trade["direction"] == "redeem" and trade["status"] == TradeStatus.success.value],
        "pre_tick_onchain_vaults": pre_tick_onchain_vaults,
        "onchain_vaults": _collect_vault_observations(receipt_web3, target_vaults, owner_address, include_availability=False),
        "queue_vault": _collect_queue_vault_observation(receipt_web3, queue_vault, owner_address),
    }
    state.other_data.save(cycle, LAGOON_ALWAYS_OPEN_LIVE_E2E_OBSERVATION_KEY, observation)


def _force_settle_controller_vaults(
    receipt_web3,
    cycle: int,
    target_vaults: list[str],
    asset_managers: list[str],
    trading_strategy_modules: list[str],
    vault_cache: dict[str, object],
) -> None:
    for index, (target_vault, schedule) in enumerate(zip(target_vaults, _target_vault_scenario())):
        deposit_settle_cycle = schedule["deposit_settle_cycle"]
        redemption_settle_cycle = schedule["redemption_settle_cycle"]
        if cycle not in (deposit_settle_cycle, redemption_settle_cycle):
            continue
        vault = _load_lagoon_vault(receipt_web3, target_vault, trading_strategy_modules[index], vault_cache)
        valuation = vault.underlying_token.fetch_balance_of(vault.safe_address)
        valuation_tx_hash = vault.post_new_valuation(valuation).transact({"from": asset_managers[index], "gas": 15_000_000})
        assert_transaction_success_with_explanation(receipt_web3, valuation_tx_hash)
        settle_tx_hash = vault.settle_via_trading_strategy_module(valuation).transact({"from": asset_managers[index], "gas": 15_000_000})
        assert_transaction_success_with_explanation(receipt_web3, settle_tx_hash, tracing=True)


def before_strategy_tick(input: StrategyTickHookInput) -> None:
    """Record local Lagoon vault state before the normal live strategy tick."""
    receipt_web3 = _controller_web3()
    target_vaults = _target_vault_addresses()
    _CONTROLLER_STATE["pre_tick_onchain_vaults"][input.cycle] = _collect_vault_observations(
        receipt_web3,
        target_vaults,
        _controller_owner_address(),
        include_availability=True,
    )


def after_strategy_tick(input: StrategyTickHookInput) -> None:
    """Record state JSON evidence and settle local Lagoon requests after the tick."""
    receipt_web3 = _controller_web3()
    target_vaults = _target_vault_addresses()
    vault_cache = _CONTROLLER_STATE["vault_cache"]
    owner_address = _controller_owner_address()
    _record_observation(
        input.state,
        input.cycle,
        input.timestamp,
        target_vaults,
        _queue_vault_address(),
        receipt_web3,
        owner_address,
        _CONTROLLER_STATE["pre_tick_onchain_vaults"].get(input.cycle, {}),
    )
    input.store.sync(input.state)
    _force_settle_controller_vaults(
        receipt_web3,
        input.cycle,
        target_vaults,
        _target_vault_asset_managers(),
        _target_vault_trading_strategy_modules(),
        vault_cache,
    )
    input.store.sync(input.state)


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
        pair.other_data["vault_role"] = "lagoon_target"
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
    """Run scheduled target-vault deposits/redemptions followed by pre-redemption queue-vault cash management."""
    state = input.state
    strategy_universe = input.strategy_universe
    target_pairs = [_get_pair(strategy_universe, address) for address in _target_vault_addresses()]
    queue_pair = _get_pair(strategy_universe, _queue_vault_address())
    position_manager = input.get_position_manager()

    yield_rules = _make_yield_rules(queue_pair)
    venue_pair_ids = queue_vault_pair_ids(yield_rules)
    alpha_model = PhaseAwareAlphaModel(input.timestamp, cycle=input.cycle, venue_pair_ids=venue_pair_ids)

    target_vault_scenario = _target_vault_scenario()
    for pair, schedule in zip(target_pairs, target_vault_scenario):
        if input.cycle < schedule["deposit_open_cycle"]:
            alpha_model.set_signal(pair, 0.0)
        elif input.cycle < schedule["redemption_open_cycle"]:
            if _has_target_trade(state, pair):
                position = position_manager.get_current_position_for_pair(pair, pending=True)
                current_value = 0.0 if position is None else position.get_value() + position.get_vault_settlement_pending_value()
                alpha_model.set_signal(pair, current_value)
                signal = alpha_model.raw_signals.get(pair.internal_id)
                if signal is not None:
                    signal.carry_forward_position = True
            else:
                alpha_model.set_signal(pair, 1.0)
        else:
            alpha_model.set_signal(pair, 0.0)

    alpha_model.carry_forward_non_redeemable_positions(position_manager)
    deployable = state.portfolio.get_total_equity() * Parameters.allocation

    alpha_model.select_top_signals(count=len(target_pairs))
    alpha_model.assign_weights(method=weight_passthrouh)
    alpha_model.normalise_weights(investable_equity=deployable, max_weight=1.0 / len(target_pairs))
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

    first_redemption_open_cycle = min(schedule["redemption_open_cycle"] for schedule in target_vault_scenario)
    if input.cycle < first_redemption_open_cycle:
        # This live e2e proves queue-vault parking before target deposits; re-parking
        # redeemed cash is covered by YieldManager tests and would lengthen the Anvil run.
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
