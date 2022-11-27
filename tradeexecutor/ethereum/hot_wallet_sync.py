"""Handling of stablecoin deposits as strategy reserves."""

import datetime
from typing import List

from eth_typing import HexAddress
from web3 import Web3

from tradeexecutor.ethereum.wallet import sync_reserves
from tradeexecutor.state.sync import apply_sync_events
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.identifier import AssetIdentifier


class EthereumHotWalletReserveSyncer:
    """Checks any Ethereum address for changes in the portfolio that may have happened outside the drawing.

    - Withdrawals

    - Deposits

    - Rebases

    - Interest payments

    .. note ::

        Private key nonce sync is handled by
        :py:func:`tradeexecutor.ethereum.UniswapV2ExecutionModel.uniswap_v2_execution.initialize`.
    """

    def __init__(self, web3: Web3, wallet_address: HexAddress):
        self.web3 = web3
        self.wallet_address = wallet_address

    def __call__(self, portfolio: Portfolio, ts: datetime.datetime, supported_reserves: List[AssetIdentifier]):
        events = sync_reserves(self.web3, ts, self.wallet_address, [], supported_reserves)
        apply_sync_events(portfolio, events)
        return events

