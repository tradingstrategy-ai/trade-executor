import os
import sys

from web3 import Web3
from tradeexecutor.ethereum.one_delta.analysis import analyse_one_delta_trade

if len(sys.argv) != 2:
    print("Usage: poetry run python scripts/analyse-1delta-trade.py <tx_hash>")
    sys.exit(1)

web3 = Web3(Web3.HTTPProvider(os.environ["JSON_RPC_POLYGON"]))
tx_hash = sys.argv[1]
analyse_one_delta_trade(web3, tx_hash=tx_hash)
