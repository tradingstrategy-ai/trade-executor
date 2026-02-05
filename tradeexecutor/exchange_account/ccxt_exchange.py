"""CCXT-specific account value function.

Provides the account value function for CEX exchange accounts via CCXT.
Initial implementation supports Aster futures (reading totalMarginBalance).
"""

import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Callable

from tradeexecutor.state.identifier import TradingPairIdentifier

if TYPE_CHECKING:
    import ccxt


logger = logging.getLogger(__name__)


def create_ccxt_exchange(
    exchange_id: str,
    options: dict | None = None,
    sandbox: bool = False,
) -> "ccxt.Exchange":
    """Create and configure a CCXT exchange instance.

    Example:

    .. code-block:: python

        from tradeexecutor.exchange_account.ccxt_exchange import create_ccxt_exchange

        exchange = create_ccxt_exchange("aster", {
            "apiKey": "your-api-key",
            "secret": "your-api-secret",
        })

    :param exchange_id:
        CCXT exchange identifier (e.g. ``"aster"``, ``"binance"``, ``"bybit"``)
    :param options:
        CCXT exchange constructor options dict.
        Passed directly to the exchange class constructor.
        Typically includes ``apiKey``, ``secret``, ``password``,
        and exchange-specific settings.
    :param sandbox:
        Whether to use the exchange's sandbox/testnet mode
    :return:
        Configured CCXT exchange instance
    :raises ValueError:
        If ``exchange_id`` is not a known CCXT exchange
    """
    import ccxt as ccxt_lib

    exchange_class = getattr(ccxt_lib, exchange_id, None)
    if exchange_class is None:
        raise ValueError(
            f"Unknown CCXT exchange: '{exchange_id}'. "
            f"See https://docs.ccxt.com/#/exchanges for supported exchanges."
        )

    config = dict(options) if options else {}
    exchange = exchange_class(config)

    if sandbox:
        exchange.set_sandbox_mode(True)

    return exchange


def aster_total_equity(exchange: "ccxt.Exchange") -> Decimal:
    """Get total account equity from Aster futures.

    Calls the Aster ``/fapi/v4/account`` endpoint directly via CCXT's
    implicit API method and extracts ``totalMarginBalance``.

    ``totalMarginBalance`` = ``totalWalletBalance`` + ``totalUnrealizedProfit``

    :param exchange:
        Authenticated CCXT Aster exchange instance
    :return:
        Total margin balance in USD as Decimal
    :raises KeyError:
        If ``totalMarginBalance`` field is not in the response
    """
    response = exchange.fapiPrivateGetV4Account()
    total_margin_balance = response.get("totalMarginBalance")
    if total_margin_balance is None:
        raise KeyError(
            f"totalMarginBalance not found in Aster account response. "
            f"Available keys: {list(response.keys())}"
        )
    value = Decimal(str(total_margin_balance))
    logger.debug("Aster total margin balance: $%.2f", value)
    return value


def create_ccxt_account_value_func(
    exchanges: dict[str, "ccxt.Exchange"],
    value_extractor: "Callable[[ccxt.Exchange], Decimal] | None" = None,
) -> Callable[[TradingPairIdentifier], Decimal]:
    """Create CCXT-specific account value function.

    The returned function queries a CCXT exchange for the total account
    value in USD, using a pluggable value extractor function.

    Example:

    .. code-block:: python

        from tradeexecutor.exchange_account.ccxt_exchange import (
            create_ccxt_exchange,
            create_ccxt_account_value_func,
            aster_total_equity,
        )

        exchange = create_ccxt_exchange("aster", {
            "apiKey": os.environ["ASTER_API_KEY"],
            "secret": os.environ["ASTER_API_SECRET"],
        })

        exchanges = {"aster_main": exchange}
        account_value_func = create_ccxt_account_value_func(exchanges)

        # Use with pricing model
        from tradeexecutor.exchange_account.pricing import ExchangeAccountPricingModel
        pricing = ExchangeAccountPricingModel(account_value_func)

    :param exchanges:
        Dict mapping ``ccxt_account_id`` -> authenticated CCXT exchange instance.
        The key matches the ``ccxt_account_id`` in the pair's ``other_data``.
    :param value_extractor:
        Function that takes a CCXT exchange instance and returns
        the total account value in USD as Decimal.
        Defaults to :func:`aster_total_equity`.
    :return:
        Function that takes a TradingPairIdentifier and returns account value in USD
    """
    if value_extractor is None:
        value_extractor = aster_total_equity

    def get_ccxt_account_value(pair: TradingPairIdentifier) -> Decimal:
        """Get account value from CCXT exchange.

        :param pair:
            Exchange account trading pair with CCXT metadata in other_data
        :return:
            Total account value in USD
        :raises KeyError:
            If ccxt_account_id not in exchanges dict
        """
        assert pair.is_exchange_account(), f"Not an exchange account pair: {pair}"
        assert pair.get_exchange_account_protocol() == "ccxt", \
            f"Not a CCXT pair: {pair.get_exchange_account_protocol()}"

        config = pair.get_exchange_account_config()
        account_id = config.get("ccxt_account_id")
        if account_id is None:
            raise ValueError(
                f"No ccxt_account_id in pair other_data: {pair.other_data}"
            )

        exchange = exchanges.get(account_id)
        if exchange is None:
            raise KeyError(
                f"No exchange for ccxt_account_id '{account_id}'. "
                f"Available: {list(exchanges.keys())}"
            )

        try:
            total_value = value_extractor(exchange)
            logger.debug(
                "CCXT account '%s' (%s) value: $%.2f",
                account_id,
                exchange.id,
                total_value,
            )
            return total_value
        except Exception as e:
            logger.error(
                "Failed to get CCXT account value for '%s' (%s): %s",
                account_id,
                exchange.id,
                e,
            )
            raise

    return get_ccxt_account_value
