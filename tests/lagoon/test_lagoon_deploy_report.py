from types import SimpleNamespace

from tradeexecutor.ethereum.lagoon import deploy_report as deploy_report_module
from tradeexecutor.ethereum.lagoon.deploy_report import generate_multichain_deployment_report


def test_generate_multichain_deployment_report_uses_deployment_chain_id(monkeypatch):
    captured: dict[str, object] = {}

    class DummyWeb3:
        def __init__(self, chain_id: int):
            self.eth = SimpleNamespace(chain_id=chain_id)

    chain_web3 = {
        42161: DummyWeb3(42161),
        8453: DummyWeb3(8453),
    }

    deployments = {
        "arbitrum": SimpleNamespace(
            chain_id=42161,
            vault=object(),
            trading_strategy_module=SimpleNamespace(address="0x1000000000000000000000000000000000000001"),
            is_satellite=False,
            get_deployment_data=lambda: {
                "Deployer": "0x2000000000000000000000000000000000000002",
                "Safe": "0x3000000000000000000000000000000000000003",
                "Vault": "0x4000000000000000000000000000000000000004",
                "Trading strategy module": "0x1000000000000000000000000000000000000001",
                "Asset manager": "0x5000000000000000000000000000000000000005",
                "Asset managers": "0x5000000000000000000000000000000000000005",
                "Valuation manager": "0x5000000000000000000000000000000000000005",
                "Share token symbol": "CSV",
                "Underlying symbol": "USDC",
                "Performance fee": "20 %",
                "Management fee": "2 %",
                "Block number": "123",
            },
        ),
        "base": SimpleNamespace(
            chain_id=8453,
            vault=object(),
            trading_strategy_module=SimpleNamespace(address="0x6000000000000000000000000000000000000006"),
            is_satellite=True,
            get_deployment_data=lambda: {
                "Deployer": "0x2000000000000000000000000000000000000002",
                "Safe": "0x3000000000000000000000000000000000000003",
                "Vault": "N/A (satellite chain)",
                "Trading strategy module": "0x6000000000000000000000000000000000000006",
                "Asset manager": "0x5000000000000000000000000000000000000005",
                "Asset managers": "0x5000000000000000000000000000000000000005",
                "Valuation manager": "0x5000000000000000000000000000000000000005",
                "Performance fee": "20 %",
                "Management fee": "2 %",
                "Block number": "124",
            },
        ),
    }

    def fake_fetch_guard_config_events(**kwargs):
        captured["module_addresses_override"] = kwargs["module_addresses_override"]
        return {}, kwargs["module_addresses_override"], {}

    monkeypatch.setattr(deploy_report_module, "fetch_guard_config_events", fake_fetch_guard_config_events)
    monkeypatch.setattr(deploy_report_module, "build_multichain_guard_config", lambda events, safe_address, module_addresses: "config")
    monkeypatch.setattr(deploy_report_module, "format_guard_config_report", lambda **kwargs: "unicode")
    monkeypatch.setattr(deploy_report_module, "format_guard_config_markdown", lambda **kwargs: "guard-markdown")

    unicode_report, markdown_report = generate_multichain_deployment_report(
        safe_address="0x3000000000000000000000000000000000000003",
        chain_web3=chain_web3,
        deployment_result=SimpleNamespace(deployments=deployments),
        simulate=True,
        from_block={42161: 100, 8453: 100},
    )

    assert captured["module_addresses_override"] == {
        42161: "0x1000000000000000000000000000000000000001",
        8453: "0x6000000000000000000000000000000000000006",
    }
    assert unicode_report == "unicode"
    assert markdown_report.startswith("# Deployment report")
