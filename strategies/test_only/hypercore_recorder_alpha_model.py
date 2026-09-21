"""Record a live HyperCore vault universe and its AlphaModel selection.

``tests/cli/test_cli_hypercore_strategy_input_recorder.py`` loads this module
through the real ``start`` command. The example exists to prove that recorder
output explains both sides of a live vault decision: the constructed universe
and the smaller set that the strategy selects from it.

The strategy follows the live Hyper-AI data path closely. It asks the curator
for real HyperCore vaults, resolves their Trading Strategy metadata, downloads
their daily price and TVL history, calculates the same TVL and age-ramp inputs
used by ``hyper-ai-test.py``, and feeds eligible vaults to ``AlphaModel``. It
stops before position sizing and trade generation because this integration test
is about decision inputs, not moving assets from the test hot wallet.

This is intentionally a checked-in strategy instead of Python source generated
inside a test. Developers can run and inspect it like any other strategy, and
the black-box test exercises normal strategy discovery without replacing CLI,
runner, universe, scheduler, or recorder methods.
"""

import datetime
import logging

import pandas as pd
from eth_defi.token import USDC_NATIVE_TOKEN
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.curator.curator import is_quarantined
from tradeexecutor.curator.hyperliquid_vault_universe import build_hyperliquid_vault_universe
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorDependencyResolver, IndicatorSet, IndicatorSource
from tradeexecutor.strategy.pandas_trader.indicator_decorator import IndicatorRegistry
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.recorder import record_decision
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse,
    load_partial_data,
    load_vault_universe_with_metadata,
)
from tradeexecutor.strategy.weighting import weight_equal


logger = logging.getLogger(__name__)

trading_strategy_engine_version = "0.5"

PREFERRED_STABLECOIN = AssetIdentifier(
    chain_id=ChainId.hyperliquid.value,
    address=USDC_NATIVE_TOKEN[ChainId.hyperliquid.value].lower(),
    token_symbol="USDC",
    decimals=6,
)


class Parameters:
    """Configure a short live-data recorder exercise without placing trades.

    Strategy-module discovery converts this conventional class to
    :class:`StrategyParameters`. It is not a dataclass because class-style
    parameters are the public v0.5 strategy-module contract.
    """

    #: Use one-second scheduling so the CLI integration test completes quickly.
    cycle_duration = CycleDuration.cycle_1s
    #: HyperCore vault observations are published as daily series.
    candle_time_bucket = TimeBucket.d1
    #: Cross-chain universe identity used by the production Hyper-AI strategy.
    chain_id = ChainId.cross_chain
    #: HyperEVM supplies the reserve asset and live Web3 connection.
    primary_chain_id = ChainId.hyperliquid
    #: Let the generic router configure HyperCore vault support normally.
    routing = TradeRouting.default
    #: Load enough history to calculate a meaningful vault-age ramp.
    required_history_period = datetime.timedelta(days=120)
    #: Exercise automatic input capture around every live decision callback.
    record_strategy_inputs = True
    #: Match Hyper-AI's live vault transaction slippage assumption.
    slippage_tolerance = 0.006

    #: Curator floor applied before the constructed universe is downloaded.
    min_tvl = 7_500
    #: Keep the live download bounded while retaining more candidates than slots.
    universe_size = 12
    #: Select fewer vaults than the curator returns so selection is observable.
    max_assets_in_portfolio = 3
    #: Match the simple survivor-first test strategy's age-ramp signal.
    age_ramp_period = 0.75

    #: Required strategy-module fields, unused by this live-only example.
    initial_cash = 10_000
    backtest_start = datetime.datetime(2025, 1, 1)
    backtest_end = datetime.datetime(2025, 1, 2)


def create_trading_universe(input: CreateTradingUniverseInput) -> TradingStrategyUniverse:
    """Construct the real live HyperCore vault universe recorded by the test.

    The live universe model calls this during ``start`` bootstrap. Keeping the
    same curator, metadata, and history functions as Hyper-AI is important:
    the integration test must cover the data that actually reaches a live
    decision, not a hand-built dataframe that merely resembles it.

    :param input:
        Live universe-construction context supplied by the framework.
    :return:
        HyperCore vault pairs with daily share-price and TVL history.
    """
    parameters = input.parameters or Parameters

    # First ask the production curator for a bounded subset of real vault
    # addresses. Closed vaults remain in the source set, as in Hyper-AI v8, so
    # their availability metadata is available to later decision checks.
    vaults = build_hyperliquid_vault_universe(
        min_tvl=float(parameters.min_tvl),
        top_n=int(parameters.universe_size),
        min_age=0.0,
        include_closed_vaults=True,
        printer=logger.info,
    )

    # Resolve addresses to the same metadata objects used by live vault
    # trading. These objects supply names, denomination tokens, fees, current
    # TVL, and deposit/redemption availability to the constructed universe.
    vault_universe = load_vault_universe_with_metadata(
        input.client,
        vaults=vaults,
        check_all_vaults_found=True,
    )

    # Download daily share-price and TVL frames from the production data API.
    # There are no supporting DEX pairs because this test selects only vaults.
    dataset = load_partial_data(
        client=input.client,
        time_bucket=parameters.candle_time_bucket,
        pairs=[],
        execution_context=input.execution_context,
        universe_options=input.universe_options,
        liquidity=True,
        liquidity_time_bucket=TimeBucket.d1,
        vaults=vault_universe,
        vault_history_source="trading-strategy-website",
        check_all_vaults_found=True,
    )

    # Forward filling matches live Hyper-AI: indicator lookup at an intraday
    # one-second cycle sees the most recent completed daily vault observation.
    return TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset=PREFERRED_STABLECOIN,
        forward_fill=True,
        forward_fill_until=input.timestamp,
        primary_chain=parameters.primary_chain_id,
    )


indicators = IndicatorRegistry()


@indicators.define(source=IndicatorSource.tvl)
def tvl(close: pd.Series) -> pd.Series:
    """Expose each vault's recorded daily TVL to selection and diagnostics.

    The indicator engine calls this once per vault using the liquidity close
    series. Keeping this as a named indicator lets the recorder fingerprint the
    exact series whose latest value becomes a candidate's TVL admission check.

    :param close:
        Daily TVL close series supplied by ``IndicatorSource.tvl``.
    :return:
        Unchanged TVL series used by ``decide_trades()``.
    """
    return close


@indicators.define()
def age(close: pd.Series) -> pd.Series:
    """Calculate vault age in years from its first available price sample.

    The indicator engine calls this per vault. Hyper-AI uses age to avoid
    assigning a mature signal immediately to newly observed vaults.

    :param close:
        Daily vault share-price series whose first row marks observed inception.
    :return:
        Vault age in fractional years at every source timestamp.
    """
    inception = close.index[0]
    age_years = (close.index - inception) / pd.Timedelta(days=365.25)
    return pd.Series(age_years, index=close.index)


@indicators.define(
    dependencies=(age,),
    source=IndicatorSource.dependencies_only_per_pair,
)
def age_ramp_weight(
    pair: TradingPairIdentifier,
    dependency_resolver: IndicatorDependencyResolver,
    age_ramp_period: float,
) -> pd.Series:
    """Ramp a new vault's selection signal up to one over its first months.

    The indicator dependency resolver calls this per vault after :func:`age`.
    ``decide_trades()`` uses the latest value as the raw AlphaModel signal.

    :param pair:
        Vault pair being calculated; required by the per-pair indicator API.
    :param dependency_resolver:
        Resolver used to read the matching vault's age series.
    :param age_ramp_period:
        Fractional years after which the signal reaches one.
    :return:
        Signal series clipped to the ``0.05 ... 1.0`` range.
    """
    vault_age = dependency_resolver.get_indicator_data("age", pair=pair)
    return (vault_age / age_ramp_period).clip(upper=1.0).clip(lower=0.05)


def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
) -> IndicatorSet:
    """Calculate the live TVL and age-ramp inputs before each decision.

    ``PandasTraderRunner`` calls this through the normal v0.5 indicator path.
    The recorder fingerprints the resulting series before ``decide_trades()``
    reads their latest values.

    :param timestamp:
        Live decision timestamp supplied by the indicator runner.
    :param parameters:
        Loaded strategy parameters containing the age-ramp period.
    :param strategy_universe:
        Constructed HyperCore vault universe to calculate over.
    :param execution_context:
        Live execution mode and engine configuration.
    :return:
        Indicator definitions expanded for the current universe.
    """
    return indicators.create_indicators(
        timestamp=timestamp,
        parameters=parameters,
        strategy_universe=strategy_universe,
        execution_context=execution_context,
    )


def _pair_key(pair: TradingPairIdentifier) -> str:
    """Return the stable vault address used to join recorder observations.

    :param pair:
        Constructed HyperCore vault pair.
    :return:
        Lowercase vault address shared by universe and selection records.
    """
    return str(pair.pool_address or pair.base.address).lower()


@record_decision
def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Rank live vault candidates and record why AlphaModel selected them.

    ``PandasTraderRunner.on_clock()`` invokes this for each one-second test
    cycle. The callback mirrors the candidate screening and top-signal steps in
    ``hyper-ai-test.py``. Returning no trades deliberately stops after the
    selection boundary, allowing a real hot wallet with no funds to run the
    black-box test safely.

    :param input:
        Complete live strategy input assembled by ``PandasTraderRunner``.
    :return:
        Empty trade list because this example stops after AlphaModel selection.
    """
    parameters = input.parameters
    candidates: list[dict[str, object]] = []
    alpha_model = AlphaModel(input.timestamp)

    # Walk the exact pair objects placed in the constructed universe. The
    # recorder has already captured this universe at decorator entry, so pair
    # IDs and addresses below can be joined back to its metadata and TVL frame.
    for pair in input.strategy_universe.iterate_pairs():
        address = _pair_key(pair)
        current_tvl = input.indicators.get_indicator_value("tvl", pair=pair)
        age_signal = input.indicators.get_indicator_value("age_ramp_weight", pair=pair)

        # Use the same safety gates as Hyper-AI before a signal enters the
        # AlphaModel. Keep a rejection reason for every excluded vault so the
        # recording explains absence as well as presence.
        rejection_reason = None
        if current_tvl is None or current_tvl != current_tvl:
            rejection_reason = "missing_tvl"
        elif float(current_tvl) < float(parameters.min_tvl):
            rejection_reason = "below_min_tvl"
        elif age_signal is None or age_signal != age_signal:
            rejection_reason = "missing_age_signal"
        elif not input.state.is_good_pair(pair):
            rejection_reason = "bad_pair"
        elif is_quarantined(address, input.timestamp):
            rejection_reason = "quarantined"

        eligible = rejection_reason is None
        candidate = {
            "pair_id": pair.internal_id,
            "vault_address": address,
            "vault_name": pair.get_vault_name() or pair.get_ticker(),
            "ticker": pair.get_ticker(),
            "tvl": None if current_tvl is None else float(current_tvl),
            "signal": None if age_signal is None else float(age_signal),
            "eligible": eligible,
            "rejection_reason": rejection_reason,
        }
        candidates.append(candidate)

        # Only accepted candidates become raw AlphaModel signals. This is the
        # same separation between inclusion rules and ranking used by Hyper-AI.
        if eligible:
            alpha_model.set_signal(pair, float(age_signal))

    # Perform the production AlphaModel pick and equal-weight steps. Because
    # there are more curated vaults than slots, this produces an inspectable
    # selected subset rather than recording every candidate as selected.
    alpha_model.select_top_signals(count=int(parameters.max_assets_in_portfolio))
    alpha_model.assign_weights(method=weight_equal)

    selected: list[dict[str, object]] = []
    for rank, signal in enumerate(alpha_model.get_signals_sorted_by_weight(), start=1):
        selected.append({
            "rank": rank,
            "pair_id": signal.pair.internal_id,
            "vault_address": _pair_key(signal.pair),
            "vault_name": signal.pair.get_vault_name() or signal.pair.get_ticker(),
            "ticker": signal.pair.get_ticker(),
            "signal": float(signal.signal),
            "raw_weight": float(signal.raw_weight),
        })

    # Explicit observations preserve the compact calculation outputs that do
    # not belong in executor state. The much larger universe frames and
    # indicator fingerprints were captured automatically by the decorator.
    assert input.recorder is not None
    input.recorder.record(
        "selection",
        "vault_candidates",
        candidates,
        arguments={"minimum_tvl": USDollarAmount(parameters.min_tvl)},
    )
    input.recorder.record(
        "selection",
        "selected_vaults",
        selected,
        arguments={
            "maximum_assets": int(parameters.max_assets_in_portfolio),
            "weighting": "equal",
        },
    )

    # Keep the AlphaModel in normal discardable diagnostics just like Hyper-AI.
    # This is state-owned diagnostic data and is therefore not duplicated as a
    # third recorder observation.
    input.state.visualisation.set_discardable_data("alpha_model", alpha_model)

    # No trade generation is needed to verify universe and selection capture.
    return []


name = "HyperCore recorder AlphaModel integration strategy"
short_description = "Loads live HyperCore vault data and records AlphaModel selection."
