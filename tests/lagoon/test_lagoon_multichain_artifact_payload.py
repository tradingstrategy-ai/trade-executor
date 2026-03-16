from types import SimpleNamespace

from eth_defi.cctp.whitelist import CCTPDeployment
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import (
    LagoonConfig,
    LagoonDeploymentParameters,
    WhitelistEntry,
)
from eth_defi.gmx.whitelist import GMXDeployment

from tradeexecutor.cli.commands.lagoon_deploy_vault import _build_multichain_artifact_payload


def _addr(value: int) -> str:
    return f"0x{value:040x}"


def _contract(address: str) -> SimpleNamespace:
    return SimpleNamespace(address=address)


def test_build_multichain_artifact_payload_contains_guard_report_and_config_snapshot():
    primary_asset_manager = _addr(10)
    secondary_asset_manager = _addr(11)
    safe_address = _addr(100)
    vault_address = _addr(101)
    module_address = _addr(102)

    config = LagoonConfig(
        parameters=LagoonDeploymentParameters(
            underlying=_addr(1),
            name="Example fund",
            symbol="EXAM",
            managementRate=200,
            performanceRate=2000,
        ),
        safe_owners=[_addr(2)],
        safe_threshold=1,
        asset_managers=[primary_asset_manager, secondary_asset_manager],
        uniswap_v2=SimpleNamespace(
            factory=_contract(_addr(3)),
            router=_contract(_addr(4)),
            weth=_contract(_addr(5)),
            init_code_hash="0x1234",
        ),
        uniswap_v3=SimpleNamespace(
            factory=_contract(_addr(6)),
            swap_router=_contract(_addr(7)),
            position_manager=_contract(_addr(8)),
            quoter=_contract(_addr(9)),
            weth=_contract(_addr(5)),
            quoter_v2=True,
            router_v2=False,
        ),
        aave_v3=SimpleNamespace(
            pool=_contract(_addr(12)),
            data_provider=_contract(_addr(13)),
            oracle=_contract(_addr(14)),
            ausdc=SimpleNamespace(address=_addr(15)),
        ),
        gmx_deployment=GMXDeployment(
            exchange_router=_addr(16),
            synthetics_router=_addr(17),
            order_vault=_addr(18),
            markets=[_addr(19)],
            tokens=[_addr(20)],
        ),
        cctp_deployment=CCTPDeployment(
            token_messenger=_addr(21),
            message_transmitter=_addr(22),
            token_minter=_addr(23),
            allowed_destination_domains=[3, 6],
        ),
        any_asset=True,
        etherscan_api_key="secret-api-key",
        verifier="etherscan",
        verifier_url="https://example.invalid/api",
        use_forge=True,
        erc_4626_vaults=[SimpleNamespace(vault_address=_addr(24), name="Spark vault", symbol="spUSDC")],
        hypercore_vaults=[_addr(25)],
        assets=[_addr(26)],
        safe_salt_nonce=42,
        satellite_chain=False,
    )

    deployment_data = {
        "Deployer": _addr(30),
        "Safe": safe_address,
        "Trading strategy module": module_address,
        "Asset manager": primary_asset_manager,
        "Asset managers": f"{primary_asset_manager}, {secondary_asset_manager}",
        "Valuation manager": primary_asset_manager,
        "Vault": vault_address,
        "Block number": "123",
    }

    deployment = SimpleNamespace(
        vault=SimpleNamespace(address=vault_address),
        safe_address=safe_address,
        trading_strategy_module=SimpleNamespace(address=module_address),
        asset_manager=primary_asset_manager,
        asset_managers=(primary_asset_manager, secondary_asset_manager),
        valuation_manager=primary_asset_manager,
        is_satellite=False,
        whitelisted_items=(
            WhitelistEntry("Sender", "trade-executor", primary_asset_manager),
            WhitelistEntry("Sender", "trade-executor", secondary_asset_manager),
        ),
        get_deployment_data=lambda: deployment_data,
    )

    result = SimpleNamespace(
        safe_address=safe_address,
        deployments={"arbitrum": deployment},
    )

    text_payload, json_payload = _build_multichain_artifact_payload(
        result=result,
        safe_salt_nonce=42,
        chain_configs={"arbitrum": config},
        guard_report="Guard tree output",
    )

    assert "Chain: arbitrum" in text_payload
    assert "Lagoon config" in text_payload
    assert "Any asset: True" in text_payload
    assert "Guard report" in text_payload
    assert "Guard tree output" in text_payload

    arbitrum_payload = json_payload["deployments"]["arbitrum"]
    assert json_payload["guard_report"] == "Guard tree output"
    assert arbitrum_payload["deployment_data"]["Safe"] == safe_address
    assert arbitrum_payload["config"]["asset_managers"] == [primary_asset_manager, secondary_asset_manager]
    assert arbitrum_payload["config"]["etherscan_api_key"] == "<redacted>"
    assert arbitrum_payload["config"]["uniswap_v2"]["router"] == _addr(4)
    assert arbitrum_payload["config"]["uniswap_v3"]["position_manager"] == _addr(8)
    assert arbitrum_payload["config"]["aave_v3"]["oracle"] == _addr(14)
    assert arbitrum_payload["config"]["gmx_deployment"]["markets"] == [_addr(19)]
    assert arbitrum_payload["config"]["cctp_deployment"]["allowed_destination_domains"] == [3, 6]
    assert arbitrum_payload["config"]["erc_4626_vaults"][0]["address"] == _addr(24)
    assert arbitrum_payload["whitelisted_items"][1]["address"] == secondary_asset_manager


def test_build_multichain_artifact_payload_omits_safe_salt_nonce_when_reusing_existing_safe():
    safe_address = _addr(200)
    old_guard_address = _addr(201)
    module_address = _addr(203)
    asset_manager = _addr(202)

    config = LagoonConfig(
        parameters=LagoonDeploymentParameters(
            underlying=_addr(1),
            name="Example fund",
            symbol="EXAM",
        ),
        safe_owners=[_addr(2)],
        safe_threshold=1,
        asset_managers=[asset_manager],
        guard_only=True,
        existing_safe_address=safe_address,
        existing_vault_address=_addr(203),
        satellite_chain=False,
    )

    deployment = SimpleNamespace(
        safe=SimpleNamespace(
            address=safe_address,
            retrieve_modules=lambda: [old_guard_address],
        ),
        vault=SimpleNamespace(address=_addr(204)),
        safe_address=safe_address,
        trading_strategy_module=SimpleNamespace(address=module_address),
        old_trading_strategy_module=SimpleNamespace(address=old_guard_address),
        asset_manager=asset_manager,
        asset_managers=(asset_manager,),
        valuation_manager=asset_manager,
        is_satellite=False,
        whitelisted_items=(),
        get_deployment_data=lambda: {
            "Deployer": _addr(205),
            "Safe": safe_address,
            "Trading strategy module": module_address,
            "Asset manager": asset_manager,
            "Asset managers": asset_manager,
            "Valuation manager": asset_manager,
            "Vault": _addr(204),
            "Block number": "123",
        },
    )

    result = SimpleNamespace(
        safe_address=safe_address,
        deployments={"arbitrum": deployment},
    )

    text_payload, json_payload = _build_multichain_artifact_payload(
        result=result,
        safe_salt_nonce=None,
        chain_configs={"arbitrum": config},
        guard_report="Guard configuration",
    )

    assert "Deployment mode: guard redeploy" in text_payload
    assert "Multichain Lagoon deployment\nDeployment mode: guard redeploy\nShared Safe:" in text_payload
    assert "Guard migration instructions" in text_payload
    assert f"{safe_address}.disableModule(0x0000000000000000000000000000000000000001, {old_guard_address})" in text_payload
    assert f"{safe_address}.enableModule({module_address})" in text_payload
    assert json_payload["deployment_mode"] == "guard redeploy"
    assert json_payload["safe_salt_nonce"] is None
    assert json_payload["deployments"]["arbitrum"]["deployment_mode"] == "guard redeploy"
    assert json_payload["deployments"]["arbitrum"]["guard_migration"]["old_guard_address"] == old_guard_address
