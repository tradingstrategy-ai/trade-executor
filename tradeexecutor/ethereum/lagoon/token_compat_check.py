"""Token compatibility check for Lagoon."""
from pathlib import Path

from tradeexecutor.ethereum.lagoon.execution import LagoonExecution
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

from eth_defi.lagoon.lagoon_compatibility import check_lagoon_compatibility_with_database, LagoonTokenCheckDatabase

UniswapV2Path = list[str]


def generate_lagoon_check_paths(
    strategy_universe: TradingStrategyUniverse,
) -> dict[TradingPairIdentifier, UniswapV2Path]:
    """

    :param strategy_universe:
        Cache warmed up TradingStrategyUniverse
    """

    reserve_token = strategy_universe.reserve_assets[0]

    #: TradingPairIdentifier -> LagoonCompatibilityCheckData routing path
    paths = {}

    for pair in strategy_universe.iterate_pairs():
        uniswap_v2_like = pair.exchange_name in ("pancakeswap-v2", "uniswap-v2", "quickswap")

        # Check not needed
        if not uniswap_v2_like:
            pair.other_data["lagoon_compat_check_data"] = None
            continue

        if pair.quote == reserve_token:
            path = [pair.quote.address, pair.base.address]
        else:
            path = [reserve_token.address, pair.quote.address, pair.base.address]

        paths[pair] = path

    return paths


def check_tokens_for_lagoon(
    strategy_universe: TradingStrategyUniverse,
    lagoon_execution: LagoonExecution,
    max_tokens: int | None = None,
    database_file: Path = Path.home() / ".tradingstrategy" / "token-checks" / "lagoon_token_check.pickle",
) -> LagoonTokenCheckDatabase:
    """Check tokens for Lagoon compatibility.

    - Uses `~/.tradingstrategy/token-checks` cache path
    - Update tokens untradable data in strategy universe with `TradingPairIdentifier.other_data["lagoon_compat_tradeable"]` flag

    :param strategy_universe:
        Cache warmed up TradingStrategyUniverse

    :param max_tokens:
        Limit token checks in unit tests for speed
    """

    assert isinstance(strategy_universe, TradingStrategyUniverse), f"Expected TradingStrategyUniverse instance. got {type(strategy_universe)}"
    assert isinstance(lagoon_execution, LagoonExecution), f"Expected LagoonExecution instance, got {type(lagoon_execution)}"

    path_map = generate_lagoon_check_paths(strategy_universe)
    pair_map = {k.base.address: k for k in path_map.keys()}

    web3 = lagoon_execution.web3
    vault = lagoon_execution.vault
    asset_manager_address = lagoon_execution.tx_builder.get_gas_wallet_address()

    paths = list(path_map.values())

    # Run Anvil scan
    compat_db = check_lagoon_compatibility_with_database(
        web3=web3,
        paths=paths,
        vault_address=vault.vault_address,
        trading_strategy_module_address=vault.trading_strategy_module_address,
        asset_manager_address=asset_manager_address,
        database_file=database_file,
    )

    entries = list(compat_db.report_by_token.values())

    if max_tokens:
        entries = entries[0:max_tokens]

    # Populate TradingPairIdentifier.other_data with LagoonCompatibilityCheckData in strategy universe
    for entry in entries:
        base_address = entry.path[-1]
        pair = pair_map[base_address]
        pair.other_data["lagoon_compat_check_data"] = entry
        pair.other_data["lagoon_compat_tradeable"] = entry.is_compatible()
        pair.other_data["lagoon_compat_revert_reason"] = entry.revert_reason

        if not entry.is_compatible():
            pair.set_tradeable(False, f"Lagoon check failed: {entry.revert_reason}")

    strategy_universe.other_data["lagoon_compat_check"] = compat_db.calculate_stats()

    return compat_db