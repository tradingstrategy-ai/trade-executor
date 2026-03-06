"""Guard configuration report after Lagoon vault deployment."""

import logging

from eth_typing import HexAddress
from web3 import Web3

from eth_defi.erc_4626.vault_protocol.lagoon.config_event_scanner import (
    build_multichain_guard_config,
    fetch_guard_config_events,
    format_guard_config_report,
)

logger = logging.getLogger(__name__)


def print_deployment_report(
    safe_address: HexAddress,
    module_address: HexAddress,
    web3: Web3,
    hypersync_api_key: str | None = None,
    simulate: bool = False,
    from_block: int = 0,
) -> None:
    """Print a guard configuration report after vault deployment.

    Reads guard config events from on-chain and formats them as a
    human-readable tree report.

    - For simulated (Anvil) deployments, uses ``eth_getLogs`` directly
    - For live deployments, uses HyperSync when ``hypersync_api_key`` is provided

    :param from_block:
        Block number to start scanning from.
        Pass the deployment block to avoid scanning the entire chain history.
    """

    chain_id = web3.eth.chain_id
    chain_web3 = {chain_id: web3}

    # Set up HyperSync client for live deployments
    hs_client = None
    if not simulate and hypersync_api_key:
        try:
            import hypersync
            from eth_defi.hypersync.server import get_hypersync_server

            hypersync_url = get_hypersync_server(web3)
            hs_config = hypersync.ClientConfig(url=hypersync_url, bearer_token=hypersync_api_key)
            hs_client = hypersync.HypersyncClient(hs_config)
        except ImportError:
            logger.warning("hypersync not installed — falling back to RPC for guard config scan")

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
    report = format_guard_config_report(
        config=config,
        events=events,
        chain_web3=chain_web3,
    )
    print(report)
