"""Binance Futures exchange support via CCXT.

Provides helpers to create an authenticated Binance Futures (``binanceusdm``)
CCXT exchange instance and utility functions for querying leverage, margin
balance, and open positions.

Intended to be used alongside :mod:`tradeexecutor.exchange_account.ccxt_exchange`.
"""

import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import ccxt

logger = logging.getLogger(__name__)


def create_binance_futures_exchange(
    api_key: str,
    api_secret: str,
    testnet: bool = False,
    default_leverage: int = 1,
) -> "ccxt.binanceusdm":
    """Create an authenticated Binance USD-M Futures CCXT exchange instance.

    Uses the ``binanceusdm`` CCXT driver which targets the
    ``https://fapi.binance.com`` endpoint.

    :param api_key:
        Binance Futures API key (Futures-enabled, not spot).
    :param api_secret:
        Binance Futures API secret.
    :param testnet:
        When ``True`` the testnet endpoint is used.
        See https://testnet.binancefuture.com for test credentials.
    :param default_leverage:
        Default cross-margin leverage applied when opening new positions.
        Binance accepts values 1-125 depending on the symbol notional.
    :return:
        Configured :class:`ccxt.binanceusdm` instance with markets loaded.
    :raises ImportError:
        If the ``ccxt`` package is not installed.
    """
    import ccxt as ccxt_lib

    exchange: ccxt_lib.binanceusdm = ccxt_lib.binanceusdm({
        "apiKey": api_key,
        "secret": api_secret,
        "options": {
            "defaultType": "future",
            "defaultLeverage": default_leverage,
        },
    })

    if testnet:
        exchange.set_sandbox_mode(True)

    exchange.load_markets()
    logger.info(
        "Binance Futures exchange created: testnet=%s leverage=%s",
        testnet,
        default_leverage,
    )
    return exchange


def get_futures_account_balance(exchange: "ccxt.binanceusdm") -> Decimal:
    """Return the available USDT margin balance on the Futures wallet.

    Calls ``/fapi/v2/balance`` via CCXT and returns the ``availableBalance``
    field for the USDT asset.

    :param exchange:
        An authenticated :func:`create_binance_futures_exchange` instance.
    :return:
        Available USDT balance as :class:`decimal.Decimal`.
    :raises KeyError:
        If no USDT entry is found in the account balance response.
    """
    balances = exchange.fetch_balance({"type": "future"})
    usdt = balances.get("USDT", {})
    available = usdt.get("free")
    if available is None:
        raise KeyError(
            "USDT free balance not found in Binance Futures response. "
            f"Available assets: {list(balances.keys())}"
        )
    value = Decimal(str(available))
    logger.debug("Binance Futures available USDT: %.4f", value)
    return value


def get_open_positions(exchange: "ccxt.binanceusdm") -> list:
    """Fetch all open leveraged positions on the Binance Futures account.

    Returns only positions with a non-zero ``contracts`` amount.

    :param exchange:
        An authenticated :func:`create_binance_futures_exchange` instance.
    :return:
        List of CCXT position dicts with at minimum the keys:
        ``symbol``, ``side``, ``contracts``, ``leverage``,
        ``entryPrice``, ``liquidationPrice``, ``unrealizedPnl``.
    """
    positions = exchange.fetch_positions()
    open_positions = [p for p in positions if p.get("contracts", 0) != 0]
    logger.debug("Binance Futures open positions: %d", len(open_positions))
    return open_positions


def set_leverage(exchange: "ccxt.binanceusdm", symbol: str, leverage: int) -> None:
    """Set the margin leverage for a specific trading pair.

    Calls ``POST /fapi/v1/leverage`` via CCXT.
    Must be called before placing an order if the desired leverage
    differs from the account default.

    :param exchange:
        An authenticated :func:`create_binance_futures_exchange` instance.
    :param symbol:
        CCXT unified symbol, e.g. ``"BTC/USDT:USDT"``.
    :param leverage:
        Integer leverage factor (1-125). Binance caps the maximum
        depending on notional position size.
    :raises ccxt.BadSymbol:
        If the symbol is not found on Binance Futures.
    """
    exchange.set_leverage(leverage, symbol)
    logger.info("Set leverage=%d for %s", leverage, symbol)


def get_liquidation_price(position: dict) -> Optional[Decimal]:
    """Extract the liquidation price from a CCXT position dict.

    :param position:
        Position dict as returned by :func:`get_open_positions`.
    :return:
        Liquidation price as :class:`decimal.Decimal`, or ``None``
        if the field is absent or zero.
    """
    raw = position.get("liquidationPrice")
    if raw is None or float(raw) == 0:
        return None
    return Decimal(str(raw))
