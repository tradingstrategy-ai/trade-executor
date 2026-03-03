"""GMX-specific exchange account support.

Provides the exchange account pair creation and account value function
for GMX perpetuals positions traded through a Lagoon vault.

GMX positions are on-chain (unlike Derive which uses an off-chain API),
so the account value function reads position data directly from the
GMX Reader contract via :class:`~eth_defi.gmx.core.open_positions.GetOpenPositions`.
"""

import logging
from decimal import Decimal
from typing import Callable

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind

logger = logging.getLogger(__name__)


def create_gmx_exchange_account_pair(
    quote: AssetIdentifier,
    safe_address: str,
    is_testnet: bool = False,
) -> TradingPairIdentifier:
    """Create a TradingPairIdentifier for a GMX exchange account.

    Builds the pair with correct ``kind``, ``exchange_name``, and ``other_data``
    fields needed by the sync, pricing, and valuation pipeline.

    The base asset is a synthetic ``GMX-ACCOUNT`` token. The Safe address
    (which holds GMX positions) is stored in ``pool_address`` and
    ``exchange_address`` for use by the account value function at query time.

    Example::

        from tradeexecutor.exchange_account.gmx import create_gmx_exchange_account_pair

        pair = create_gmx_exchange_account_pair(
            quote=usdc,
            safe_address="0xAbC...",
        )

    :param quote:
        Reserve / quote asset (e.g. ``USDC``).
    :param safe_address:
        Address of the Safe multisig that holds GMX positions.
    :param is_testnet:
        Whether this targets a testnet deployment.
    :return:
        Fully configured exchange account pair.
    """
    # Synthetic asset representing the GMX account value
    # 694D = "GM" in ASCII-ish hex
    base = AssetIdentifier(
        chain_id=quote.chain_id,
        address="0x000000000000000000000000000000000000694D",
        token_symbol="GMX-ACCOUNT",
        decimals=6,
    )

    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=safe_address,
        exchange_address=safe_address,
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="GMX",
        other_data={
            "exchange_protocol": "gmx",
            "exchange_is_testnet": is_testnet,
        },
    )


def create_gmx_account_value_func(
    web3,
) -> Callable[[TradingPairIdentifier], Decimal]:
    """Create GMX-specific account value function.

    The returned function queries the GMX Reader contract for all open
    positions held by the Safe address, and returns the total equity
    (collateral + unrealised PnL) across all positions.

    Free USDC sitting in the Safe is tracked separately by Lagoon
    treasury sync, so this function only returns the value locked
    in GMX positions to avoid double counting.

    Example::

        from web3 import Web3
        from tradeexecutor.exchange_account.gmx import create_gmx_account_value_func

        web3 = Web3(Web3.HTTPProvider("https://arb1.arbitrum.io/rpc"))
        value_func = create_gmx_account_value_func(web3)

        # Use with pricing model
        value = value_func(gmx_pair)

    :param web3:
        Web3 instance connected to the chain where GMX positions live
        (typically Arbitrum).
    :return:
        Function that takes a TradingPairIdentifier and returns
        the total GMX position equity in USD.
    """

    def get_gmx_account_value(pair: TradingPairIdentifier) -> Decimal:
        """Get GMX account value for the given pair.

        Reads all open positions from the GMX Reader contract for the
        Safe address stored in ``pair.pool_address``, then sums the
        equity of each position:
        ``equity = collateral_usd * (1 + percent_profit / 100)``.

        :param pair:
            Exchange account trading pair with GMX metadata in other_data.
        :return:
            Total position equity in USD, or ``Decimal(0)`` if no
            open positions.
        """
        # Lazy imports to avoid loading GMX modules at startup
        from eth_defi.gmx.config import GMXConfig
        from eth_defi.gmx.core.open_positions import GetOpenPositions

        assert pair.is_exchange_account(), f"Not an exchange account pair: {pair}"
        assert pair.get_exchange_account_protocol() == "gmx", \
            f"Not a GMX pair: {pair.get_exchange_account_protocol()}"

        safe_address = pair.pool_address
        if not safe_address:
            raise ValueError(f"No pool_address (Safe address) in pair: {pair}")

        try:
            config = GMXConfig(web3=web3)
            positions_manager = GetOpenPositions(config)
            positions = positions_manager.get_data(safe_address)

            if not positions:
                logger.debug("No open GMX positions for %s", safe_address)
                return Decimal(0)

            total_equity = Decimal(0)
            for key, pos in positions.items():
                collateral_usd = pos.get("initial_collateral_amount_usd", 0)
                percent_profit = pos.get("percent_profit", 0)
                # equity = collateral * (1 + pnl%)
                equity = Decimal(str(collateral_usd)) * (1 + Decimal(str(percent_profit)) / 100)
                total_equity += equity
                logger.debug(
                    "GMX position %s: collateral=$%.2f, pnl=%.2f%%, equity=$%.2f",
                    key, collateral_usd, percent_profit, equity,
                )

            logger.debug(
                "GMX account %s total equity: $%.2f (%d position(s))",
                safe_address, total_equity, len(positions),
            )
            return total_equity

        except Exception as e:
            logger.error(
                "Failed to get GMX account value for %s: %s",
                safe_address, e,
            )
            raise

    return get_gmx_account_value
