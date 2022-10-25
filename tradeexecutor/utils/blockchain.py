"""Blockchain related utilities."""
import datetime

from web3 import Web3


def get_latest_block_timestamp(web3: Web3) -> datetime.datetime:
    """Get the latest block timestamp.

    :return:
        Timezone naive datetime
    """
    last_block = web3.eth.get_block("latest")
    ts_str = last_block["timestamp"]

    # Depending on middleware, response might be converted or not
    if type(ts_str) == str:
        ts = int(ts_str, 16)
    else:
        ts = ts_str

    return datetime.datetime.utcfromtimestamp(ts)