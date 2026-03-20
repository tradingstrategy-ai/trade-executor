import re

from tradingstrategy.chain import ChainId
from tradingstrategy.pair import DEXPair

from tradeexecutor.state.identifier import TradingPairIdentifier


def parse_pair_data(s: str) -> list[str] | tuple:
    """Extract pair data from string.

    :param s:
        String in the format of: [(chain_id, exchange_slug, base_token, quote_token, fee)])],

        where rate is optional.

    :raises ValueError:
        If the string is not in the correct format.

    :return:
        Tuple of (chain_id, exchange_slug, base_token, quote_token, fee)"""

    try:
        # Extract the tuple
        tuple_str = re.search(r'\((.*?)\)', s)[1]

        # Split elements and remove leading/trailing whitespaces
        elements = [e.strip() for e in tuple_str.split(',')]

        if len(elements) not in {2, 4, 5}:
            raise ValueError()

        if len(elements) == 2:
            # Short format: (chain_id, exchange_slug)
            chain_id = getattr(ChainId, elements[0].split('.')[-1])
            address = elements[1].strip('"')
            assert address.startswith('0x'), f"Not an address: {address}"
            return [chain_id, address]

        # Process elements
        chain_id = getattr(ChainId, elements[0].split('.')[-1])
        exchange_slug = elements[1].strip('"')
        base_token = elements[2].strip('"')
        quote_token = elements[3].strip('"')
        fee = float(elements[4]) if len(elements) > 4 else None

    except:
        raise ValueError(f'Invalid pair data: {s}. Tuple must be in the format of: (chain_id, exchange_slug, base_token, quote_token, fee), where fee is optional')

    return (chain_id, exchange_slug, base_token, quote_token, fee)


def construct_identifier_from_pair(pair: DEXPair) -> str:
    """Construct pair identifier from pair data.

    :param pair:
        Pair data as DEXPair.

    :return:
        Pair identifier string."""

    assert isinstance(pair, DEXPair), 'Pair must be of type DEXPair'

    return f'({pair.chain_id.name}, "{pair.exchange_slug}", "{pair.base_token_symbol}", "{pair.quote_token_symbol}", {pair.fee/10_000})'


def construct_identifier_from_trading_pair(pair: TradingPairIdentifier) -> str:
    """Construct a CLI-compatible pair identifier string from a TradingPairIdentifier.

    Unlike :py:func:`construct_identifier_from_pair` which works with DEXPair,
    this works with the executor's internal TradingPairIdentifier.

    :param pair:
        Trading pair identifier.

    :return:
        Pair identifier string suitable for ``--pair`` CLI option.
    """
    chain_name = ChainId(pair.base.chain_id).name
    fee = pair.fee if pair.fee else 0
    return f'({chain_name}, "{pair.exchange_name}", "{pair.base.token_symbol}", "{pair.quote.token_symbol}", {fee})'
