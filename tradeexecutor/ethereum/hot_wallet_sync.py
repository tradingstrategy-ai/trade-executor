import datetime
from typing import List

from eth_typing import HexAddress
from web3 import Web3

from tradeexecutor.ethereum.wallet import sync_reserves, sync_portfolio
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.identifier import AssetIdentifier


class EthereumHotWalletReserveSyncer:
    """Checks any Ethereum address for changes in the portfolio that may have happened outside the drawing.

    - Withdrawals
    - Deposits
    - Rebases
    - Interest payments

    """

    def __init__(self, web3: Web3, wallet_address: HexAddress):
        self.web3 = web3
        self.wallet_address = wallet_address

    def __call__(self, portfolio: Portfolio, ts: datetime.datetime, supported_reserves: List[AssetIdentifier]):
        events = sync_reserves(self.web3, ts, self.wallet_address, [], supported_reserves)
        sync_portfolio(portfolio, events)
        return events

