"""Backfill share price records on statistics when it was missing."""
import logging
import datetime

from eth_defi.erc_4626.vault import ERC4626Vault
from eth_defi.provider.named import get_provider_name
from eth_defi.provider.quicknode import estimate_block_number_for_timestamp_by_quicknode
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore

logger = logging.getLogger(__name__)


def retrofit_share_price(
    state: State,
    vault: ERC4626Vault,
    max_time: datetime.datetime | None = None,
    store: JSONFileStore | None = None,
) -> int:
    """Backfill missing share price data in state from Lagoon onchain data.

    = Quicknode RPC required (for now) to get timestamps

    :param max_time:
        Unit test loop limiter

    :return:
        Number of historical portfolio statistics entries backfilled.

    """
    web3 = vault.web3
    statistics = state.stats
    updates = 0

    logger.info("Retrofitting share price data for vault %s", vault)

    name = get_provider_name(web3.provider)
    assert "quiknode" in name, f"QuickNode provider required, got {name}"

    for portfolio_stats in statistics.portfolio:

        if portfolio_stats.share_count:
            # ALready filled
            continue

        timestamp = portfolio_stats.calculated_at

        if timestamp > max_time:
            break

        block_number_reply = estimate_block_number_for_timestamp_by_quicknode(
            web3,
            timestamp,
        )

        block_number = block_number_reply.block_number

        share_count = vault.fetch_total_supply(block_number)
        share_price = vault.fetch_share_price(block_number)

        logger.info(
            "Backfilling share price data. Shares %f, price %f %s at %s (%s)",
            share_count,
            share_price,
            vault.denomination_token,
            timestamp,
            f"{block_number:,}"
        )

        portfolio_stats.share_price_usd = share_price
        portfolio_stats.share_count = share_count

        updates += 1

        if updates % 100 == 0 and store:
            store.sync(state)

    logger.info("Share price updated for %d portfolio entries", updates)

    return updates
