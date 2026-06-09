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
    4. Confirm share token balance and total supply are logged for the default chain.
    """
    from tradeexecutor.cli.commands import check_wallet as check_wallet_module

    SHARE_TOKEN_ADDRESS = "0x3333333333333333333333333333333333333333"
    SHARE_TOKEN_SYMBOL = "lagVault"
    SHARE_BALANCE = 42
    SHARE_TOTAL_SUPPLY = 1000

    class FakeShareToken:
        def __init__(self):
            self.address = SHARE_TOKEN_ADDRESS
            self.symbol = SHARE_TOKEN_SYMBOL
            self.contract = SimpleNamespace(
                functions=SimpleNamespace(
                    totalSupply=lambda: SimpleNamespace(call=lambda: SHARE_TOTAL_SUPPLY),
                ),
            )

        def fetch_balance_of(self, address: str) -> int:
            if address == HOT_WALLET_ADDRESS:
                return SHARE_BALANCE
            return 0

        def convert_to_decimals(self, balance: int) -> int:
            return balance

    class FakeLagoonVaultSyncModel:
        def __init__(self, hot_wallet_address: str, safe_address: str, vault_address: str):
            self._hot_wallet_address = hot_wallet_address
            self._safe_address = safe_address
            self.vault_address = vault_address
            self.vault = SimpleNamespace(share_token=FakeShareToken())

        def get_hot_wallet(self):
            return SimpleNamespace(address=self._hot_wallet_address)

        def get_token_storage_address(self) -> str:
            return self._safe_address

    class FakeExecutionModel:
        satellite_vaults = {}

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
            self.connections = {
                ChainId.hyperliquid: SimpleNamespace(
                    eth=SimpleNamespace(
                        chain_id=ChainId.hyperliquid.value,
                        block_number=30_000_000,
                        get_balance=lambda address: 0,
                    )
                ),
                ChainId.base: SimpleNamespace(
                    eth=SimpleNamespace(
                        chain_id=ChainId.base.value,
                        block_number=40_000_000,
                        get_balance=lambda address: 0,
                    )
                ),
            }

        def has_chain_configured(self) -> bool:
            return True

        def set_default_chain(self, chain_id: ChainId) -> None:
            self.default_chain_id = chain_id

        def check_default_chain_id(self) -> None:
            return None

        def get_connection(self, chain_id: ChainId):
            return self.connections[chain_id]

        def get_default(self):
            return self.connections[self.default_chain_id]

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
    BASE_SAFE_ADDRESS = "0x2222222222222222222222222222222222222222"
    VAULT_ADDRESS = "0x1111111111111111111111111111111111111111"
    USDC_ADDRESS = "0xb88339cb7199b77e23db6e890353e22632ba630f"
    BASE_USDC_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

    reserve_asset = AssetIdentifier(
        chain_id=ChainId.hyperliquid.value,
        address=USDC_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )
    base_reserve_asset = AssetIdentifier(
        chain_id=ChainId.base.value,
        address=BASE_USDC_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )
    cctp_pair = SimpleNamespace(
        base=base_reserve_asset,
        quote=reserve_asset,
        is_cctp_bridge=lambda: True,
    )
    pair_iterator = iter([cctp_pair])
    fake_universe = SimpleNamespace(
        reserve_assets=[reserve_asset],
        cross_chain=True,
        data_universe=SimpleNamespace(
            pairs=[],
            chains={ChainId.hyperliquid, ChainId.base},
        ),
        iterate_pairs=pair_iterator,
    )
    fake_execution_model = FakeExecutionModel()
    fake_execution_model.satellite_vaults = {
        ChainId.base.value: SimpleNamespace(safe_address=BASE_SAFE_ADDRESS),
    }
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
        "LOG_LEVEL": "disabled",
        "ASSET_MANAGEMENT_MODE": "lagoon",
    }

    # 1. Patch the CLI dependencies so `check-wallet` runs against `hyper-ai-test.py` with a fake Lagoon sync model.
    monkeypatch.setattr(check_wallet_module, "LagoonVaultSyncModel", FakeLagoonVaultSyncModel)
    monkeypatch.setattr(check_wallet_module, "prepare_executor_id", lambda id, strategy_file: "test-executor")
    monkeypatch.setattr(
        check_wallet_module,
        "prepare_cache_and_token_cache",
        lambda id, cache_path, unit_testing=False: (tmp_path, object()),
    )
    monkeypatch.setattr(check_wallet_module, "create_web3_config", lambda **kwargs: FakeWeb3Config())
    monkeypatch.setattr(check_wallet_module.Client, "create_live_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(check_wallet_module, "call_create_trading_universe", lambda *args, **kwargs: fake_universe)
    monkeypatch.setattr(
        check_wallet_module,
        "create_execution_and_sync_model",
        lambda **kwargs: (fake_execution_model, fake_sync_model, None, None),
    )
    monkeypatch.setattr(
        check_wallet_module,
        "make_factory_from_strategy_mod",
        lambda mod: lambda **kwargs: SimpleNamespace(runner=FakeRunner()),
    )
    monkeypatch.setattr(check_wallet_module, "fetch_erc20_details", lambda web3, address, cache=None: FakeTokenDetails(address))
    monkeypatch.setattr(
        check_wallet_module,
        "fetch_erc20_balances_by_token_list",
        lambda web3, address, tokens: {token: 11 for token in tokens},
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
    assert any("-" * 80 in message for message in messages)
    assert any("HyperEVM (chain id 999)" in message for message in messages)
    assert any("Base (chain id 8453)" in message for message in messages)
    assert any(f"Vault address is {VAULT_ADDRESS}" in message for message in messages)
    assert any(f"Safe address is {SAFE_ADDRESS}" in message for message in messages)
    assert any(f"Safe address is {BASE_SAFE_ADDRESS}" in message for message in messages)
    assert any("Hot wallet reserve balance of USD Coin" in message for message in messages)
    assert any("Vault reserve balance of USD Coin" in message for message in messages)
    assert any(f"Vault reserve balance of USD Coin ({USDC_ADDRESS})" in message for message in messages)
    assert any(f"Safe reserve balance of USD Coin ({BASE_USDC_ADDRESS.lower()})" in message for message in messages)

    # 4. Confirm share token balance and total supply are logged for the default chain.
    assert any(f"Share token: {SHARE_TOKEN_SYMBOL} ({SHARE_TOKEN_ADDRESS})" in message for message in messages)
    assert any(f"Asset manager share balance: {SHARE_BALANCE} {SHARE_TOKEN_SYMBOL}" in message for message in messages)
    assert any(f"Total share supply: {SHARE_TOTAL_SUPPLY} {SHARE_TOKEN_SYMBOL}" in message for message in messages)


def test_cli_check_wallet_logs_unclaimed_lagoon_deposits_and_redemptions(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Test `check-wallet` warns about unclaimed Lagoon vault deposits and redemptions.

    1. Patch the CLI so the Lagoon vault has a vault_contract with maxDeposit > 0, maxRedeem > 0, and pendingRedeemRequest > 0.
    2. Run check-wallet and capture log output.
    3. Confirm UNCLAIMED DEPOSIT, UNCLAIMED REDEMPTION, and PENDING REDEMPTION warnings appear.
    """
    from tradeexecutor.cli.commands import check_wallet as check_wallet_module

    SHARE_TOKEN_ADDRESS = "0x3333333333333333333333333333333333333333"
    SHARE_TOKEN_SYMBOL = "lagVault"
    SHARE_BALANCE = 42
    SHARE_TOTAL_SUPPLY = 1000
    UNCLAIMED_DEPOSIT_RAW = 10_000_000
    UNCLAIMED_SHARES_RAW = 500
    PENDING_SHARES_RAW = 200

    class FakeDenominationToken:
        def __init__(self):
            self.symbol = "USDC"

        def convert_to_decimals(self, balance: int) -> int:
            return balance

    class FakeShareToken:
        def __init__(self):
            self.address = SHARE_TOKEN_ADDRESS
            self.symbol = SHARE_TOKEN_SYMBOL
            self.contract = SimpleNamespace(
                functions=SimpleNamespace(
                    totalSupply=lambda: SimpleNamespace(call=lambda: SHARE_TOTAL_SUPPLY),
                ),
            )

        def fetch_balance_of(self, address: str) -> int:
            if address == HOT_WALLET_ADDRESS:
                return SHARE_BALANCE
            return 0

        def convert_to_decimals(self, balance: int) -> int:
            return balance

    fake_share_token = FakeShareToken()
    fake_denomination_token = FakeDenominationToken()
    fake_vault_contract = SimpleNamespace(
        functions=SimpleNamespace(
            maxDeposit=lambda address: SimpleNamespace(call=lambda: UNCLAIMED_DEPOSIT_RAW),
            maxRedeem=lambda address: SimpleNamespace(call=lambda: UNCLAIMED_SHARES_RAW),
            pendingRedeemRequest=lambda request_id, address: SimpleNamespace(call=lambda: PENDING_SHARES_RAW),
        ),
    )

    class FakeLagoonVaultSyncModel:
        def __init__(self, hot_wallet_address: str, safe_address: str, vault_address: str):
            self._hot_wallet_address = hot_wallet_address
            self._safe_address = safe_address
            self.vault_address = vault_address
            self.vault = SimpleNamespace(
                share_token=fake_share_token,
                denomination_token=fake_denomination_token,
                vault_contract=fake_vault_contract,
            )

        def get_hot_wallet(self):
            return SimpleNamespace(address=self._hot_wallet_address)

        def get_token_storage_address(self) -> str:
            return self._safe_address

    class FakeExecutionModel:
        satellite_vaults = {}

        def preflight_check(self) -> None:
            return None

    class FakeRoutingModel:
        def perform_preflight_checks_and_logging(self, pairs) -> None:
            return None

    class FakeRunner:
        def __init__(self):
            self.routing_model = FakeRoutingModel()

        def setup_routing(self, universe):
            return None, SimpleNamespace(), SimpleNamespace()

    class FakeWeb3Config:
        def __init__(self):
            self.default_chain_id = None
            self.connections = {
                ChainId.hyperliquid: SimpleNamespace(
                    eth=SimpleNamespace(
                        chain_id=ChainId.hyperliquid.value,
                        block_number=30_000_000,
                        get_balance=lambda address: 0,
                    )
                ),
            }

        def has_chain_configured(self) -> bool:
            return True

        def set_default_chain(self, chain_id: ChainId) -> None:
            self.default_chain_id = chain_id

        def check_default_chain_id(self) -> None:
            return None

        def get_connection(self, chain_id: ChainId):
            return self.connections[chain_id]

        def get_default(self):
            return self.connections[self.default_chain_id]

        def close(self) -> None:
            return None

    class FakeTokenDetails:
        def __init__(self, address: str):
            self.name = "USD Coin"
            self.address = address
            self.symbol = "USDC"

        def fetch_balance_of(self, address: str) -> int:
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
        cross_chain=False,
        data_universe=SimpleNamespace(
            pairs=[],
            chains={ChainId.hyperliquid},
        ),
        iterate_pairs=iter([]),
    )
    fake_execution_model = FakeExecutionModel()
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
        "LOG_LEVEL": "disabled",
        "ASSET_MANAGEMENT_MODE": "lagoon",
    }

    # 1. Patch the CLI so the Lagoon vault has a vault_contract with maxRedeem > 0 and pendingRedeemRequest > 0.
    monkeypatch.setattr(check_wallet_module, "LagoonVaultSyncModel", FakeLagoonVaultSyncModel)
    monkeypatch.setattr(check_wallet_module, "prepare_executor_id", lambda id, strategy_file: "test-executor")
    monkeypatch.setattr(
        check_wallet_module,
        "prepare_cache_and_token_cache",
        lambda id, cache_path, unit_testing=False: (tmp_path, object()),
    )
    monkeypatch.setattr(check_wallet_module, "create_web3_config", lambda **kwargs: FakeWeb3Config())
    monkeypatch.setattr(check_wallet_module.Client, "create_live_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(check_wallet_module, "call_create_trading_universe", lambda *args, **kwargs: fake_universe)
    monkeypatch.setattr(
        check_wallet_module,
        "create_execution_and_sync_model",
        lambda **kwargs: (fake_execution_model, fake_sync_model, None, None),
    )
    monkeypatch.setattr(
        check_wallet_module,
        "make_factory_from_strategy_mod",
        lambda mod: lambda **kwargs: SimpleNamespace(runner=FakeRunner()),
    )
    monkeypatch.setattr(check_wallet_module, "fetch_erc20_details", lambda web3, address, cache=None: FakeTokenDetails(address))
    monkeypatch.setattr(
        check_wallet_module,
        "fetch_erc20_balances_by_token_list",
        lambda web3, address, tokens: {token: 0 for token in tokens},
    )

    runner = CliRunner()
    caplog.set_level("WARNING")

    # 2. Run check-wallet and capture log output.
    result = runner.invoke(app, ["check-wallet"], env=environment)

    if result.exception:
        raise result.exception
    assert result.exit_code == 0, result.stdout

    # 3. Confirm UNCLAIMED DEPOSIT, UNCLAIMED REDEMPTION, and PENDING REDEMPTION warnings appear.
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        f"UNCLAIMED DEPOSIT: {UNCLAIMED_DEPOSIT_RAW} USDC" in message
        for message in messages
    ), f"Expected unclaimed deposit warning not found in: {messages}"
    assert any(
        f"UNCLAIMED REDEMPTION: {UNCLAIMED_SHARES_RAW} {SHARE_TOKEN_SYMBOL}" in message
        for message in messages
    ), f"Expected unclaimed redemption warning not found in: {messages}"
    assert any(
        f"PENDING REDEMPTION: {PENDING_SHARES_RAW} {SHARE_TOKEN_SYMBOL}" in message
        for message in messages
    ), f"Expected pending redemption warning not found in: {messages}"
