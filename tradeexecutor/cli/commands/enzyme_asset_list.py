"""enzyme-asset-list CLi command."""
import datetime
import json
import logging
import sys
from typing import Optional, cast

from eth_defi.provider.broken_provider import get_block_tip_latency
from typer import Option

from eth_defi.chainlink.round_data import fetch_chainlink_round_data
from eth_defi.enzyme.deployment import POLYGON_DEPLOYMENT, EnzymeDeployment
from eth_defi.enzyme.price_feed import fetch_price_feeds, fetch_updated_price_feed
from eth_defi.event_reader.multithread import MultithreadEventReader
from eth_defi.event_reader.progress_update import PrintProgressUpdate, TQDMProgressUpdate
from eth_defi.token import fetch_erc20_details
from tradingstrategy.chain import ChainId
from web3 import HTTPProvider

from tradeexecutor.cli.bootstrap import create_web3_config
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.cli.version_info import VersionInfo
from tradeexecutor.ethereum.enzyme.asset import EnzymeAsset


@app.command()
def enzyme_asset_list(
    log_level: str = shared_options.log_level,
    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_base: Optional[str] = shared_options.json_rpc_base,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: Optional[str] = shared_options.json_rpc_anvil,
    end_block: Optional[int] = Option(None, envvar="END_BLOCK", help="End block to scan. If not given default to the latest block.")

):
    """Print out JSON list of supported Enzyme assets on a chain."""

    logger = setup_logging(log_level)
    # SaaS JSON-RPC API throttling warnings
    logging.getLogger("eth_defi.middleware").setLevel(logging.ERROR)  #  Encountered JSON-RPC retryable error 429 Client Error: Too Many Requests for url:
    logging.getLogger("futureproof").setLevel(logging.ERROR)  #  Extra noise

    web3config = create_web3_config(
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum, json_rpc_base=json_rpc_base, 
        json_rpc_anvil=json_rpc_anvil,
        json_rpc_arbitrum=json_rpc_arbitrum,
    )

    if not web3config.has_any_connection():
        raise RuntimeError("Live trading requires that you pass JSON-RPC connection to one of the networks")

    web3config.choose_single_chain()

    web3 = web3config.get_default()
    provider = cast(HTTPProvider, web3.provider)
    chain_id = ChainId(web3.eth.chain_id)

    logger.info("Connected to chain %s", chain_id.name)

    # No other supported Enzyme deployments
    match chain_id:
        case ChainId.ethereum:
            raise NotImplementedError("Not supported yet")
        case ChainId.polygon:
            deployment_info = POLYGON_DEPLOYMENT
        case _:
            raise NotImplementedError("Not supported yet")

    assert chain_id in (ChainId.ethereum, ChainId.polygon), f"Unsupported {chain_id}"

    # Read Enzyme deployment from chain
    deployment = EnzymeDeployment.fetch_deployment(web3, deployment_info)
    logger.info(f"Fetched Enzyme deployment with ComptrollerLib as %s", deployment.contracts.comptroller_lib.address)

    start_block = deployment_info["deployed_at"]

    if not end_block:
        end_block = max(1, web3.eth.block_number - get_block_tip_latency(web3))

    # Set up multithreaded Polygon event reader.
    # Print progress to the console how many blocks there are left to read.
    reader = MultithreadEventReader(
        provider.endpoint_uri,
        max_threads=16,
        notify=TQDMProgressUpdate("Scanning Enzyme Asset List"),
        max_blocks_once=10_000,
        reorg_mon=None,
    )

    logger.info(f"Scanning for Enzyme price feed events {start_block:,} - {end_block:,}")

    feeds = fetch_updated_price_feed(
        deployment,
        start_block=start_block,
        end_block=end_block,
        read_events=reader,
    )

    reader.close()

    logger.info("Found %d Enzyme price feeds", len(feeds))
    logger.info("Made %d API calls", reader.get_total_api_call_counts()["total"])

    assets = []

    for feed in feeds.values():
        logger.info("Resolving token data for %s", feed.primitive_token.address)
        persistent = EnzymeAsset.convert_raw_feed(feed)
        serialised = json.loads(persistent.to_json())
        assets.append(serialised)

    logger.info("JSON copy paste:\n")
    print('\n[\n' +
        ',\n'.join(json.dumps(i) for i in assets) +
        '\n]\n')
