import datetime
from typing import List

from tradeexecutor.ethereum.wallet import sync_reserves, sync_portfolio
from tradeexecutor.state.state import AssetIdentifier, Portfolio


class EthereumHotWalletReserveSyncer:

    def __init__(self, web3: Web3, wallet_address: HexAddress):
        self.web3 = web3
        self.wallet_address = wallet_address


    def __call__(self, portfolio: Portfolio, ts: datetime.datetime, supported_reserves: List[AssetIdentifier]):
        events = sync_reserves(self.web3, ts, self.wallet_address, [], supported_reserves)
        sync_portfolio(portfolio, events)
