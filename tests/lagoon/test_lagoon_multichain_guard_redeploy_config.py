from pathlib import Path

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.ethereum.lagoon import universe_config as universe_config_module
from tradeexecutor.ethereum.lagoon.universe_config import translate_trading_universe_to_lagoon_config
from tradeexecutor.strategy.strategy_module import read_strategy_module


def test_translate_universe_to_lagoon_config_supports_multichain_guard_redeploy(monkeypatch):
    strategy_file = Path("strategies/test_only/cctp_bridge_start_test.py")
    module = read_strategy_module(strategy_file)
    universe = module.create_trading_universe(
        ts=native_datetime_utc_now(),
        client=None,
        execution_context=None,
        universe_options=None,
    )

    monkeypatch.setattr(universe_config_module, "_apply_protocol_configs", lambda **kwargs: None)

    existing_safe_address = "0x1000000000000000000000000000000000000001"
    existing_vault_address = "0x2000000000000000000000000000000000000002"
    asset_manager = "0x3000000000000000000000000000000000000003"

    configs = translate_trading_universe_to_lagoon_config(
        universe=universe,
        chain_web3={
            "arbitrum": object(),
            "base": object(),
        },
        asset_manager=asset_manager,
        safe_owners=[asset_manager],
        safe_threshold=1,
        safe_salt_nonce=42,
        any_asset=True,
        guard_only=True,
        existing_safe_address=existing_safe_address,
        existing_vault_address=existing_vault_address,
    )

    assert configs["arbitrum"].satellite_chain is False
    assert configs["arbitrum"].guard_only is True
    assert configs["arbitrum"].existing_safe_address == existing_safe_address
    assert configs["arbitrum"].existing_vault_address == existing_vault_address

    assert configs["base"].satellite_chain is True
    assert configs["base"].guard_only is True
    assert configs["base"].existing_safe_address == existing_safe_address
    assert configs["base"].existing_vault_address is None
