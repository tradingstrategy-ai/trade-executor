"""Shared Hypercore writer fixtures.

These fixtures are intentionally grouped under a dedicated test package because
the Hypercore + Lagoon integration setup is large and we want later follow-up
tests to reuse the same Anvil fork and replay scaffolding.
"""

from __future__ import annotations

import datetime
import importlib.util
import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from web3 import Web3

from eth_defi.erc_4626.vault_protocol.lagoon.deployment import (
    LagoonAutomatedDeployment,
    LagoonDeploymentParameters,
    deploy_automated_lagoon_vault,
)
from eth_defi.hotwallet import HotWallet
from eth_defi.hyperliquid.core_writer import CORE_DEPOSIT_WALLET, CORE_WRITER_ADDRESS
from eth_defi.hyperliquid.testing import fund_erc20_on_anvil, setup_anvil_hypercore_mocks
from eth_defi.provider.anvil import AnvilLaunch, fork_network_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDC_NATIVE_TOKEN, TokenDetails, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.lagoon.execution import LagoonExecution
from tradeexecutor.ethereum.lagoon.tx import LagoonTransactionBuilder
from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.ethereum.vault.hypercore_vault import (
    HLP_VAULT_ADDRESS,
    create_hypercore_vault_pair,
)
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.generic.generic_valuation import GenericValuation
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse,
    create_pair_universe_from_code,
    translate_token,
)
from tradeexecutor.testing.hypercore_replay import HypercoreDailyMetricsReplay
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType, ExchangeUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


logger = logging.getLogger(__name__)

JSON_RPC_HYPERLIQUID = os.environ.get("JSON_RPC_HYPERLIQUID")


@dataclass
class FakeIndicators:
    """Tiny indicator adapter for exercising ``strategies/test_only/hyper-ai-test.py`` in tests."""

    values: dict[tuple[str, int | None], object]

    def get_indicator_value(
        self,
        name: str,
        pair=None,
        na_conversion=True,
        parameters=None,
    ):
        del na_conversion
        del parameters
        pair_id = None if pair is None else pair.internal_id
        return self.values.get((name, pair_id))


@pytest.fixture()
def replay_vault_address() -> str:
    return HLP_VAULT_ADDRESS["mainnet"]


@pytest.fixture()
def hypercore_daily_metrics_frame(replay_vault_address: str) -> pd.DataFrame:
    """Small real daily-metrics sample for HLP on Hyperliquid mainnet."""
    return pd.DataFrame(
        [
            {
                "vault_address": replay_vault_address,
                "date": datetime.date(2026, 1, 7),
                "tvl": 301023173.834363,
                "cumulative_pnl": 118271038.524363,
            },
            {
                "vault_address": replay_vault_address,
                "date": datetime.date(2026, 1, 21),
                "tvl": 269004391.733933,
                "cumulative_pnl": 119191482.033933,
            },
            {
                "vault_address": replay_vault_address,
                "date": datetime.date(2026, 2, 3),
                "tvl": 308585706.340077,
                "cumulative_pnl": 137908924.100077,
            },
        ]
    )


@pytest.fixture()
def hypercore_replay_source(
    replay_vault_address: str,
    hypercore_daily_metrics_frame: pd.DataFrame,
) -> HypercoreDailyMetricsReplay:
    return HypercoreDailyMetricsReplay.from_single_vault_dataframe(
        replay_vault_address,
        hypercore_daily_metrics_frame,
        lockup_expired_after=datetime.datetime(2026, 2, 1),
    )


@pytest.fixture()
def hypercore_usdc_asset() -> AssetIdentifier:
    return AssetIdentifier(
        chain_id=999,
        address=USDC_NATIVE_TOKEN[999],
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def hypercore_vault_pair(
    hypercore_usdc_asset: AssetIdentifier,
    replay_vault_address: str,
):
    return create_hypercore_vault_pair(
        quote=hypercore_usdc_asset,
        vault_address=replay_vault_address,
    )


@pytest.fixture()
def hypercore_strategy_universe(
    hypercore_vault_pair,
) -> TradingStrategyUniverse:
    exchange_universe = ExchangeUniverse(
        exchanges={
            1: Exchange(
                chain_id=ChainId(999),
                chain_slug="hyperliquid",
                exchange_id=1,
                exchange_slug="hypercore",
                address=hypercore_vault_pair.exchange_address,
                exchange_type=ExchangeType.erc_4626_vault,
                pair_count=1,
            ),
        }
    )

    pair_universe = create_pair_universe_from_code(
        ChainId.hypercore,
        [hypercore_vault_pair],
    )
    pair_universe.exchange_universe = exchange_universe

    universe = Universe(
        chains={ChainId(999)},
        time_bucket=TimeBucket.not_applicable,
        exchange_universe=pair_universe.exchange_universe,
        pairs=pair_universe,
    )

    reserve_token = pair_universe.get_token(hypercore_vault_pair.quote.address)
    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets={translate_token(reserve_token)},
    )


@pytest.fixture()
def hyper_ai_strategy_module():
    strategy_path = Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "hyper-ai-test.py"
    spec = importlib.util.spec_from_file_location("hyper_ai_strategy", strategy_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def make_fake_indicators():
    def _make(values: dict[tuple[str, int | None], object]) -> FakeIndicators:
        return FakeIndicators(values=values)

    return _make


@pytest.fixture()
def anvil_hyperevm() -> AnvilLaunch:
    if not JSON_RPC_HYPERLIQUID:
        pytest.skip("JSON_RPC_HYPERLIQUID environment variable required")

    launch = fork_network_anvil(JSON_RPC_HYPERLIQUID, gas_limit=30_000_000)
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def web3_hyperevm(anvil_hyperevm: AnvilLaunch) -> Web3:
    web3 = create_multi_provider_web3(
        anvil_hyperevm.json_rpc_url,
        default_http_timeout=(3, 250.0),
    )
    assert web3.eth.chain_id == 999
    return web3


@pytest.fixture()
def hypercore_usdc_token(web3_hyperevm: Web3) -> TokenDetails:
    return fetch_erc20_details(web3_hyperevm, USDC_NATIVE_TOKEN[999], chain_id=999)


@pytest.fixture()
def asset_manager(web3_hyperevm: Web3) -> HotWallet:
    wallet = HotWallet.create_for_testing(web3_hyperevm, eth_amount=5)
    wallet.sync_nonce(web3_hyperevm)
    setup_anvil_hypercore_mocks(web3_hyperevm, wallet.address)
    return wallet


@pytest.fixture()
def multisig_owners(web3_hyperevm: Web3) -> list[str]:
    return [web3_hyperevm.eth.accounts[1], web3_hyperevm.eth.accounts[2]]


@pytest.fixture()
def depositor(
    web3_hyperevm: Web3,
    hypercore_usdc_token: TokenDetails,
) -> str:
    account = web3_hyperevm.eth.accounts[6]
    web3_hyperevm.provider.make_request("anvil_setBalance", [account, hex(10 * 10**18)])
    fund_erc20_on_anvil(
        web3_hyperevm,
        hypercore_usdc_token.address,
        account,
        hypercore_usdc_token.convert_to_raw(Decimal("1000")),
    )
    return account


@pytest.fixture()
def automated_hypercore_lagoon_vault(
    web3_hyperevm: Web3,
    asset_manager: HotWallet,
    multisig_owners: list[str],
    replay_vault_address: str,
) -> LagoonAutomatedDeployment:
    parameters = LagoonDeploymentParameters(
        underlying=USDC_NATIVE_TOKEN[999],
        name="Hypercore replay test",
        symbol="HCRT",
    )

    return deploy_automated_lagoon_vault(
        web3=web3_hyperevm,
        deployer=asset_manager,
        asset_manager=asset_manager.address,
        parameters=parameters,
        safe_owners=multisig_owners,
        safe_threshold=2,
        any_asset=True,
    )


@pytest.fixture(autouse=False)
def _whitelist_hypercore_on_lagoon_guard(
    web3_hyperevm: Web3,
    automated_hypercore_lagoon_vault: LagoonAutomatedDeployment,
    replay_vault_address: str,
) -> None:
    """Match the stable Hypercore guard-test setup from eth_defi."""
    module = automated_hypercore_lagoon_vault.trading_strategy_module
    safe_address = automated_hypercore_lagoon_vault.safe.address

    web3_hyperevm.provider.make_request("anvil_impersonateAccount", [safe_address])
    web3_hyperevm.provider.make_request("anvil_setBalance", [safe_address, hex(10 * 10**18)])

    tx_hash = module.functions.whitelistCoreWriter(
        Web3.to_checksum_address(CORE_WRITER_ADDRESS),
        Web3.to_checksum_address(CORE_DEPOSIT_WALLET[999]),
        "Hypercore vault trading",
    ).transact({"from": safe_address})
    assert_transaction_success_with_explanation(web3_hyperevm, tx_hash)

    tx_hash = module.functions.whitelistHypercoreVault(
        Web3.to_checksum_address(replay_vault_address),
        "Replay Hypercore vault",
    ).transact({"from": safe_address})
    assert_transaction_success_with_explanation(web3_hyperevm, tx_hash)

    tx_hash = module.functions.whitelistToken(
        Web3.to_checksum_address(USDC_NATIVE_TOKEN[999]),
        "USDC for Hypercore bridging",
    ).transact({"from": safe_address})
    assert_transaction_success_with_explanation(web3_hyperevm, tx_hash)

    web3_hyperevm.provider.make_request("anvil_stopImpersonatingAccount", [safe_address])


@pytest.fixture()
def hypercore_sync_model(
    automated_hypercore_lagoon_vault: LagoonAutomatedDeployment,
    asset_manager: HotWallet,
) -> LagoonVaultSyncModel:
    return LagoonVaultSyncModel(
        vault=automated_hypercore_lagoon_vault.vault,
        hot_wallet=asset_manager,
        unit_testing=True,
    )


@pytest.fixture()
def hypercore_execution_model(
    web3_hyperevm: Web3,
    automated_hypercore_lagoon_vault: LagoonAutomatedDeployment,
    _whitelist_hypercore_on_lagoon_guard: None,
    asset_manager: HotWallet,
    hypercore_replay_source: HypercoreDailyMetricsReplay,
) -> LagoonExecution:
    execution_model = LagoonExecution(
        vault=automated_hypercore_lagoon_vault.vault,
        tx_builder=LagoonTransactionBuilder(automated_hypercore_lagoon_vault.vault, asset_manager),
        mainnet_fork=True,
        confirmation_block_count=0,
    )
    execution_model.hypercore_market_data_source = hypercore_replay_source
    return execution_model


@pytest.fixture()
def hypercore_pair_configurator(
    web3_hyperevm: Web3,
    hypercore_strategy_universe: TradingStrategyUniverse,
    hypercore_execution_model: LagoonExecution,
) -> EthereumPairConfigurator:
    return EthereumPairConfigurator(
        web3_hyperevm,
        hypercore_strategy_universe,
        execution_model=hypercore_execution_model,
    )


@pytest.fixture()
def hypercore_routing_model(
    hypercore_execution_model: LagoonExecution,
    hypercore_strategy_universe: TradingStrategyUniverse,
) -> GenericRouting:
    return hypercore_execution_model.create_default_routing_model(hypercore_strategy_universe)


@pytest.fixture()
def hypercore_pricing_model(
    hypercore_pair_configurator: EthereumPairConfigurator,
) -> GenericPricing:
    return GenericPricing(hypercore_pair_configurator)


@pytest.fixture()
def hypercore_valuation_model(
    hypercore_pair_configurator: EthereumPairConfigurator,
) -> GenericValuation:
    return GenericValuation(hypercore_pair_configurator)


@pytest.fixture()
def deposited_hypercore_vault_state(
    web3_hyperevm: Web3,
    automated_hypercore_lagoon_vault: LagoonAutomatedDeployment,
    depositor: str,
    hypercore_usdc_token: TokenDetails,
    hypercore_sync_model: LagoonVaultSyncModel,
    hypercore_strategy_universe: TradingStrategyUniverse,
) -> tuple[object, State]:
    state = State()
    reserve_asset = hypercore_strategy_universe.get_reserve_asset()
    hypercore_sync_model.sync_initial(
        state,
        reserve_asset=reserve_asset,
        reserve_token_price=1.0,
    )

    deposit_amount = Decimal("399")
    vault = automated_hypercore_lagoon_vault.vault
    raw_amount = hypercore_usdc_token.convert_to_raw(deposit_amount)

    tx_hash = hypercore_usdc_token.approve(vault.address, deposit_amount).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3_hyperevm, tx_hash)

    tx_hash = vault.request_deposit(depositor, raw_amount).transact({"from": depositor, "gas": 1_000_000})
    assert_transaction_success_with_explanation(web3_hyperevm, tx_hash)

    events = hypercore_sync_model.sync_treasury(
        datetime.datetime(2026, 1, 21),
        state,
        supported_reserves=[reserve_asset],
        post_valuation=True,
    )
    assert len(events) == 1

    return vault, state


@pytest.fixture()
def hypercore_state_with_safe_reserves(
    web3_hyperevm: Web3,
    automated_hypercore_lagoon_vault: LagoonAutomatedDeployment,
    _whitelist_hypercore_on_lagoon_guard: None,
    hypercore_strategy_universe: TradingStrategyUniverse,
    hypercore_usdc_token: TokenDetails,
) -> tuple[object, State]:
    """Fund the Lagoon Safe directly and mirror that balance into state reserves.

    This avoids the Lagoon treasury-sync path that currently hangs on the
    HyperEVM Anvil fork, while still letting us exercise Hypercore routing and
    settlement through the real Lagoon trading module.
    """
    state = State()
    reserve_asset = hypercore_strategy_universe.get_reserve_asset()
    reserve = state.portfolio.initialise_reserves(
        reserve_asset,
        reserve_token_price=1.0,
    )

    reserve_amount = Decimal("399")
    reserve.quantity = reserve_amount

    fund_erc20_on_anvil(
        web3_hyperevm,
        hypercore_usdc_token.address,
        automated_hypercore_lagoon_vault.vault.safe_address,
        hypercore_usdc_token.convert_to_raw(reserve_amount),
    )

    return automated_hypercore_lagoon_vault.vault, state
