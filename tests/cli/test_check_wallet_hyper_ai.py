"""CLI coverage for Hyper AI wallet checks."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tradeexecutor.cli.main import app
from tradeexecutor.state.identifier import AssetIdentifier
from tradingstrategy.chain import ChainId


@pytest.mark.timeout(300)
@pytest.mark.skipif(
    os.environ.get("TRADING_STRATEGY_API_KEY") is None or os.environ.get("JSON_RPC_HYPERLIQUID") is None,
    reason="Set TRADING_STRATEGY_API_KEY and JSON_RPC_HYPERLIQUID environment variables to run this test",
)
def test_cli_check_wallet_hyper_ai_test_strategy(
    tmp_path: Path,
) -> None:
    """Test `check-wallet` accepts a cross-chain strategy with a primary chain override.

    1. Configure the CLI to load `hyper-ai-test.py` with a live Trading Strategy client and Hyperliquid RPC.
    2. Run `check-wallet` through the Typer CLI entry point.
    3. Confirm the command exits successfully instead of treating `cross_chain` as the default RPC chain.
    """
    strategy_file = Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "hyper-ai-test.py"
    environment = {
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "STRATEGY_FILE": str(strategy_file),
        "CACHE_PATH": str(tmp_path),
        "JSON_RPC_HYPERLIQUID": os.environ["JSON_RPC_HYPERLIQUID"],
        "PRIVATE_KEY": "0x111e53aed5e777996f26b4bdb89300bbc05b84743f32028c41be7193c0fe0b83",
        "MIN_GAS_BALANCE": "0",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
    }

    runner = CliRunner()

    # 1. Configure the CLI to load `hyper-ai-test.py` with a live Trading Strategy client and Hyperliquid RPC.
    # 2. Run `check-wallet` through the Typer CLI entry point.
    result = runner.invoke(app, ["check-wallet"], env=environment)

    # 3. Confirm the command exits successfully instead of treating `cross_chain` as the default RPC chain.
    if result.exception:
        raise result.exception
    assert result.exit_code == 0, result.stdout


def test_cli_check_wallet_logs_hot_wallet_and_vault_reserve_balances(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Test `check-wallet` logs reserve balances for both the hot wallet and the vault.

    1. Patch the CLI dependencies so `check-wallet` runs against `hyper-ai-test.py` with a fake Lagoon sync model.
    2. Run the CLI command and capture its log output.
    3. Confirm the reserve token balance is logged separately for the hot wallet and the vault.
    """
    from tradeexecutor.cli.commands import check_wallet as check_wallet_module

    class FakeLagoonVaultSyncModel:
        def __init__(self, hot_wallet_address: str, safe_address: str, vault_address: str):
            self._hot_wallet_address = hot_wallet_address
            self._safe_address = safe_address
            self.vault_address = vault_address

        def get_hot_wallet(self):
            return SimpleNamespace(address=self._hot_wallet_address)

        def get_token_storage_address(self) -> str:
            return self._safe_address

    class FakeExecutionModel:
        def preflight_check(self) -> None:
            return None

    class FakeRoutingModel:
        def perform_preflight_checks_and_logging(self, pairs) -> None:
            return None

    class FakeRunner:
        def __init__(self):
            self.routing_model = FakeRoutingModel()

        def setup_routing(self, universe):
            return None, FakePricingModel(), FakeValuationModel()

    class FakePricingModel:
        pass

    class FakeValuationModel:
        pass

    class FakeWeb3Config:
        def __init__(self):
            self.default_chain_id = None
            self._web3 = SimpleNamespace(
                eth=SimpleNamespace(
                    chain_id=999,
                    block_number=30_000_000,
                    get_balance=lambda address: 0,
                )
            )

        def has_chain_configured(self) -> bool:
            return True

        def set_default_chain(self, chain_id: ChainId) -> None:
            self.default_chain_id = chain_id

        def check_default_chain_id(self) -> None:
            return None

        def get_default(self):
            return self._web3

        def close(self) -> None:
            return None

    class FakeTokenDetails:
        def __init__(self, address: str):
            self.name = "USD Coin"
            self.address = address
            self.symbol = "USDC"

        def fetch_balance_of(self, address: str) -> int:
            if address == HOT_WALLET_ADDRESS:
                return 5
            if address == SAFE_ADDRESS:
                return 11
            return 0

        def convert_to_decimals(self, balance: int) -> int:
            return balance

    HOT_WALLET_ADDRESS = "0xFA093b13C04c5E16e7a2A75d1279C58Df1Fbac62"
    SAFE_ADDRESS = "0x49Be988d2090aa221586e9A51cacBA3D3A1eA087"
    VAULT_ADDRESS = "0x1111111111111111111111111111111111111111"
    USDC_ADDRESS = "0xb88339cb7199b77e23db6e890353e22632ba630f"

    reserve_asset = AssetIdentifier(
        chain_id=ChainId.hyperliquid.value,
        address=USDC_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )
    fake_universe = SimpleNamespace(
        reserve_assets=[reserve_asset],
        cross_chain=True,
        data_universe=SimpleNamespace(pairs=[]),
    )
    fake_sync_model = FakeLagoonVaultSyncModel(
        hot_wallet_address=HOT_WALLET_ADDRESS,
        safe_address=SAFE_ADDRESS,
        vault_address=VAULT_ADDRESS,
    )

    strategy_file = Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "hyper-ai-test.py"
    environment = {
        "TRADING_STRATEGY_API_KEY": "test-key",
        "STRATEGY_FILE": str(strategy_file),
        "CACHE_PATH": str(tmp_path),
        "JSON_RPC_HYPERLIQUID": "https://example-hyperliquid-rpc.invalid",
        "PRIVATE_KEY": "0x111e53aed5e777996f26b4bdb89300bbc05b84743f32028c41be7193c0fe0b83",
        "MIN_GAS_BALANCE": "0",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "warning",
        "ASSET_MANAGEMENT_MODE": "lagoon",
    }

    # 1. Patch the CLI dependencies so `check-wallet` runs against `hyper-ai-test.py` with a fake Lagoon sync model.
    monkeypatch.setattr(check_wallet_module, "LagoonVaultSyncModel", FakeLagoonVaultSyncModel)
    monkeypatch.setattr(check_wallet_module, "prepare_executor_id", lambda id, strategy_file: "test-executor")
    monkeypatch.setattr(check_wallet_module, "prepare_cache", lambda id, cache_path, unit_testing=False: tmp_path)
    monkeypatch.setattr(check_wallet_module, "create_web3_config", lambda **kwargs: FakeWeb3Config())
    monkeypatch.setattr(check_wallet_module.Client, "create_live_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(check_wallet_module, "call_create_trading_universe", lambda *args, **kwargs: fake_universe)
    monkeypatch.setattr(
        check_wallet_module,
        "create_execution_and_sync_model",
        lambda **kwargs: (FakeExecutionModel(), fake_sync_model, None, None),
    )
    monkeypatch.setattr(
        check_wallet_module,
        "make_factory_from_strategy_mod",
        lambda mod: lambda **kwargs: SimpleNamespace(runner=FakeRunner()),
    )
    monkeypatch.setattr(check_wallet_module, "prepare_token_cache", lambda *args, **kwargs: object())
    monkeypatch.setattr(check_wallet_module, "fetch_erc20_details", lambda web3, address, cache=None: FakeTokenDetails(address))
    monkeypatch.setattr(
        check_wallet_module,
        "fetch_erc20_balances_by_token_list",
        lambda web3, address, tokens: {tokens[0]: 11},
    )

    runner = CliRunner()
    caplog.set_level("INFO")

    # 2. Run the CLI command and capture its log output.
    result = runner.invoke(app, ["check-wallet"], env=environment)

    # 3. Confirm the reserve token balance is logged separately for the hot wallet and the vault.
    if result.exception:
        raise result.exception
    assert result.exit_code == 0, result.stdout
    messages = [record.getMessage() for record in caplog.records]
    assert any("Hot wallet reserve balance of USD Coin" in message for message in messages)
    assert any("Vault reserve balance of USD Coin" in message for message in messages)
