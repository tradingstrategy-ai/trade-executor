import logging
from types import SimpleNamespace

from tradeexecutor.cli.commands.lagoon_deploy_vault import _resolve_multichain_fund_metadata


def test_resolve_multichain_fund_metadata_reads_existing_vault_when_redeploying(monkeypatch):
    universe = SimpleNamespace(
        reserve_assets=[
            SimpleNamespace(chain_id=42161),
        ]
    )
    chain_web3 = {
        "arbitrum": object(),
    }

    monkeypatch.setattr(
        "tradeexecutor.cli.commands.lagoon_deploy_vault.create_vault_instance",
        lambda web3, address, features: SimpleNamespace(name="GMX AI", symbol="GMXAI"),
    )

    fund_name, fund_symbol = _resolve_multichain_fund_metadata(
        fund_name=None,
        fund_symbol=None,
        existing_vault_address="0x1000000000000000000000000000000000000001",
        universe=universe,
        chain_web3=chain_web3,
        logger=logging.getLogger("test"),
    )

    assert fund_name == "GMX AI"
    assert fund_symbol == "GMXAI"


def test_resolve_multichain_fund_metadata_keeps_explicit_values():
    universe = SimpleNamespace(
        reserve_assets=[
            SimpleNamespace(chain_id=42161),
        ]
    )

    fund_name, fund_symbol = _resolve_multichain_fund_metadata(
        fund_name="Explicit fund",
        fund_symbol="EXPL",
        existing_vault_address=None,
        universe=universe,
        chain_web3={"arbitrum": object()},
        logger=logging.getLogger("test"),
    )

    assert fund_name == "Explicit fund"
    assert fund_symbol == "EXPL"
