"""Guard configuration report after Lagoon vault deployment."""

import logging

from eth_typing import HexAddress
from web3 import Web3

from eth_defi.erc_4626.vault_protocol.lagoon.config_event_scanner import (
    build_multichain_guard_config,
    fetch_guard_config_events,
    format_guard_config_markdown,
    format_guard_config_report,
)
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import LagoonMultichainDeployment
from eth_defi.etherscan.config import get_etherscan_url

logger = logging.getLogger(__name__)


def _create_hypersync_client(web3: Web3, hypersync_api_key: str):
    """Create a HyperSync client for the given chain.

    :return:
        ``hypersync.HypersyncClient`` or ``None`` if unavailable.
    """
    try:
        import hypersync
        from eth_defi.hypersync.server import get_hypersync_server

        hypersync_url = get_hypersync_server(web3)
        hs_config = hypersync.ClientConfig(url=hypersync_url, bearer_token=hypersync_api_key)
        return hypersync.HypersyncClient(hs_config)
    except ImportError:
        logger.warning("hypersync not installed — falling back to RPC for guard config scan")
        return None


def print_deployment_report(
    safe_address: HexAddress,
    module_address: HexAddress,
    web3: Web3,
    hypersync_api_key: str | None = None,
    simulate: bool = False,
    from_block: int = 0,
) -> tuple[str, str]:
    """Print a guard configuration report after single-chain vault deployment.

    Reads guard config events from on-chain and formats them as both a
    human-readable Unicode tree (printed to stdout) and a Markdown
    document (returned for file writing).

    - For simulated (Anvil) deployments, uses ``eth_getLogs`` directly
    - For live deployments, uses HyperSync when ``hypersync_api_key`` is provided

    :param from_block:
        Block number to start scanning from.
        Pass the deployment block to avoid scanning the entire chain history.

    :return:
        Tuple of ``(unicode_report, markdown_report)``.
    """

    chain_id = web3.eth.chain_id
    chain_web3 = {chain_id: web3}

    # Set up HyperSync client for live deployments
    hs_client = None
    if not simulate and hypersync_api_key:
        hs_client = _create_hypersync_client(web3, hypersync_api_key)

    events, _module_addrs = fetch_guard_config_events(
        safe_address=safe_address,
        web3=web3,
        hypersync_client=hs_client,
        chain_web3=chain_web3,
        follow_cctp=False,
        from_block=from_block,
    )

    module_addresses = {chain_id: module_address}
    config = build_multichain_guard_config(events, safe_address, module_addresses)

    unicode_report = format_guard_config_report(
        config=config,
        events=events,
        chain_web3=chain_web3,
    )
    markdown_report = format_guard_config_markdown(
        config=config,
        events=events,
        chain_web3=chain_web3,
    )

    print(unicode_report)
    return unicode_report, markdown_report


def _format_deployment_metadata_markdown(
    deployment_result: LagoonMultichainDeployment,
) -> str:
    """Build a Markdown preamble with deployment metadata.

    :param deployment_result:
        The multichain deployment result containing per-chain data.

    :return:
        Markdown string with deployment metadata table.
    """
    lines: list[str] = []
    lines.append("# Deployment report")
    lines.append("")

    for slug, dep in deployment_result.deployments.items():
        data = dep.get_deployment_data()
        chain_id = dep.vault.w3.eth.chain_id if hasattr(dep.vault, "w3") else None

        kind = "satellite" if dep.is_satellite else "source"
        lines.append(f"## {slug.title()} ({kind})")
        lines.append("")

        def _link(addr: str) -> str:
            if chain_id and addr and addr != "N/A (satellite chain)":
                explorer_url = get_etherscan_url(chain_id)
                if explorer_url:
                    return f"[`{addr}`]({explorer_url}/address/{addr})"
            return f"`{addr}`" if addr else "-"

        def _link_address_list(value: str) -> str:
            if not value:
                return "-"
            addresses = [address.strip() for address in value.split(",") if address.strip()]
            return ", ".join(_link(address) for address in addresses)

        lines.append(f"- **Deployer**: {_link(data.get('Deployer', ''))}")
        lines.append(f"- **Safe**: {_link(data.get('Safe', ''))}")
        vault_addr = data.get("Vault", "")
        if vault_addr and vault_addr != "N/A (satellite chain)":
            lines.append(f"- **Vault**: {_link(vault_addr)}")
        lines.append(f"- **Guard module**: {_link(data.get('Trading strategy module', ''))}")
        lines.append(f"- **Primary asset manager**: {_link(data.get('Asset manager', ''))}")
        lines.append(f"- **Asset managers**: {_link_address_list(data.get('Asset managers', ''))}")
        lines.append(f"- **Lagoon valuation manager**: {_link(data.get('Valuation manager', ''))}")

        if not dep.is_satellite:
            symbol = data.get("Share token symbol", "")
            if symbol:
                lines.append(f"- **Share token**: {symbol}")
            underlying = data.get("Underlying symbol", "")
            if underlying:
                lines.append(f"- **Denomination**: {underlying}")

        perf_fee = data.get("Performance fee", "")
        mgmt_fee = data.get("Management fee", "")
        if perf_fee:
            lines.append(f"- **Performance fee**: {perf_fee}")
        if mgmt_fee:
            lines.append(f"- **Management fee**: {mgmt_fee}")

        block = data.get("Block number", "")
        if block:
            lines.append(f"- **Block number**: {block}")

        lines.append("")

    return "\n".join(lines)


def generate_multichain_deployment_report(
    safe_address: HexAddress,
    chain_web3: dict[int, Web3],
    deployment_result: LagoonMultichainDeployment,
    hypersync_api_key: str | None = None,
    simulate: bool = False,
    from_block: int | dict[int, int] = 0,
    token_cache=None,
) -> tuple[str, str]:
    """Generate multichain guard configuration report in both Unicode and Markdown.

    Performs a single multichain event scan with ``follow_cctp=True``,
    builds the guard configuration, and returns both a Unicode tree
    (printed to stdout) and a Markdown document (for file writing).

    The Markdown report includes deployment metadata (addresses, fees,
    block numbers) followed by the guard configuration tree with
    block-explorer links.

    :param safe_address:
        Deterministic Safe address shared across all chains.

    :param chain_web3:
        ``{chain_id: Web3}`` mapping for all deployed chains.

    :param deployment_result:
        The multichain deployment result containing per-chain metadata.

    :param hypersync_api_key:
        Optional HyperSync API key for production scans.

    :param simulate:
        Whether this is a simulated (Anvil) deployment.

    :param from_block:
        Starting block number(s) for the event scan.

    :param token_cache:
        Optional disk cache for token metadata.

    :return:
        Tuple of ``(unicode_report, markdown_report)``.
    """
    # Pick the first available chain as the starting point
    first_chain_id = next(iter(chain_web3))
    web3 = chain_web3[first_chain_id]

    # Set up HyperSync client for live deployments
    hs_client = None
    if not simulate and hypersync_api_key:
        hs_client = _create_hypersync_client(web3, hypersync_api_key)

    events, module_addresses = fetch_guard_config_events(
        safe_address=safe_address,
        web3=web3,
        hypersync_client=hs_client,
        chain_web3=chain_web3,
        follow_cctp=True,
        from_block=from_block,
    )

    config = build_multichain_guard_config(events, safe_address, module_addresses)

    unicode_report = format_guard_config_report(
        config=config,
        events=events,
        chain_web3=chain_web3,
        token_cache=token_cache,
    )

    # Build full Markdown: deployment metadata + guard config tree
    metadata_md = _format_deployment_metadata_markdown(deployment_result)
    guard_md = format_guard_config_markdown(
        config=config,
        events=events,
        chain_web3=chain_web3,
        token_cache=token_cache,
    )

    markdown_report = metadata_md + "\n---\n\n" + guard_md

    print(unicode_report)
    return unicode_report, markdown_report
