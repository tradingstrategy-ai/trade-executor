"""Backfill share price records on statistics when it was missing."""
import logging
from os import times

from eth_defi.erc_4626.vault import ERC4626Vault
from eth_defi.timestamp import estimate_block_number_for_timestamp_by_findblock
from tradeexecutor.state.state import State


logger = logging.getLogger(__name__)


def retrofit_share_price(
    state: State,
    vault: ERC4626Vault,
) -> int:
    """Backfill missing share price data in state from Lagoon onchain data.

    :return:
        Number of historical portfolio statistics entries backfilled.

    """
    web3 = vault.web3
    chain_id = web3.eth.chain_id
    statistics = state.stats
    updates = 0

    logger.info("Retrofitting share price data for vault %", vault)

    for portfolio_stats in statistics.portfolio:

        if portfolio_stats.share_count:
            # ALready filled
            continue

        timestamp = portfolio_stats.calculated_at
        block_number = estimate_block_number_for_timestamp_by_findblock(
            chain_id,
            timestamp,
        )

        share_count = vault.fetch_total_supply(block_number)
        share_price = vault.fetch_share_price(block_number)

        logger.info(
            "Backfilling share price data. Shares %f, price %f %s at %s (%s)",
            share_count,
            share_price,
            timestamp,
            f"{block_number:,}"
        )

        portfolio_stats.share_price_usd = share_price
        portfolio_stats.share_count = share_count

        updates += 1

    logger.info("Share price updated for %d portfolio entries", updates)

    return updates
