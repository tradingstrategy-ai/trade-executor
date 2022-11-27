"""Web3 connection configuration."""

import logging
from dataclasses import field, dataclass
from typing import Dict, Optional

from eth_defi.gas import GasPriceMethod, node_default_gas_price_strategy
from tradingstrategy.chain import ChainId
from web3 import Web3, HTTPProvider
from web3.middleware import geth_poa_middleware

from tradeexecutor.utils.url import get_url_domain

#: List of currently supportd EVM blockchains
SUPPORTED_CHAINS = [
    ChainId.ethereum,
    ChainId.avalanche,
    ChainId.polygon,
    ChainId.bsc,
]


logger = logging.getLogger(__name__)


@dataclass
class Web3Config:
    """Advanced Web3 connection manager.

    Supports multiple blockchain connections.
    """

    #: Mapping of different connections for different chains
    connections: Dict[ChainId, Web3] = field(default_factory=dict)

    #: How do we price our txs
    gas_price_method: Optional[GasPriceMethod] = None

    #: Chain id for single chain strategies
    default_chain_id: Optional[ChainId] = None

    @staticmethod
    def create_web3(url: str, gas_price_method: Optional[GasPriceMethod] = None) -> Web3:
        """Create a new Web3.py connection.

        :param url:
            JSON-RPC node URL

        :param gas_price_method:
            How do we estimate gas for a transaction
            If not given autodetect the method.
        """

        web3 = Web3(HTTPProvider(url))

        # Read numeric chain id from JSON-RPC
        chain_id = web3.eth.chain_id

        if gas_price_method is None:
            if chain_id == ChainId.ethereum.value:
                # Ethereum supports maxBaseFee method

                gas_price_method = GasPriceMethod.london
            else:
                # Other nodes have the legacy method
                #
                #   File "/Users/moo/Library/Caches/pypoetry/virtualenvs/trade-executor-8Oz1GdY1-py3.10/lib/python3.10/site-packages/web3/contract.py", line 1672, in build_transaction_for_function
                #     prepared_transaction = fill_transaction_defaults(web3, prepared_transaction)
                #   File "cytoolz/functoolz.pyx", line 249, in cytoolz.functoolz.curry.__call__
                #   File "/Users/moo/Library/Caches/pypoetry/virtualenvs/trade-executor-8Oz1GdY1-py3.10/lib/python3.10/site-packages/web3/_utils/transactions.py", line 114, in fill_transaction_defaults
                #     default_val = default_getter(web3, transaction)
                #   File "/Users/moo/Library/Caches/pypoetry/virtualenvs/trade-executor-8Oz1GdY1-py3.10/lib/python3.10/site-packages/web3/_utils/transactions.py", line 64, in <lambda>
                #     web3.eth.max_priority_fee + (2 * web3.eth.get_block('latest')['baseFeePerGas'])
                #   File "/Users/moo/Library/Caches/pypoetry/virtualenvs/trade-executor-8Oz1GdY1-py3.10/lib/python3.10/site-packages/web3/datastructures.py", line 51, in __getitem__
                #     return self.__dict__[key]  # type: ignore
                # KeyError: 'baseFeePerGas'

                gas_price_method = GasPriceMethod.legacy

        assert isinstance(gas_price_method, GasPriceMethod)

        logger.info("Connected to chain id: %d, using gas price method %s", chain_id, gas_price_method.name)

        # London is the default method
        if gas_price_method == GasPriceMethod.legacy:
            logger.info("Setting up gas price middleware for Web3")
            web3.eth.set_gas_price_strategy(node_default_gas_price_strategy)

        # Set POA middleware if needed
        if chain_id in (ChainId.bsc.value, ChainId.polygon.value):
            logger.info("Using proof-of-authority web3 middleware")
            web3.middleware_onion.inject(geth_poa_middleware, layer=0)

        return web3

    def close(self):
        """Close all connections."""
        # Web3.py does not offer close
        pass

    def set_default_chain(self, chain_id: ChainId):
        """Set the default chain our strategy runs on.

        Most strategies are single chain strategies.
        Set the chain id we expect these strategies to run on.
        """
        assert isinstance(chain_id, ChainId), f"Got {chain_id}"
        self.default_chain_id = chain_id

    def get_connection(self, chain_id: ChainId) -> Optional[Web3]:
        """Get a connection to a specific network."""
        return self.connections[chain_id]

    def get_default(self) -> Web3:
        """Getst the default connection.

        Assumes exactly 1 node connection available.
        """
        assert self.default_chain_id
        try:
            return self.connections[self.default_chain_id]
        except KeyError:
            raise RuntimeError(f"We haev {self.default_chain_id.name} configured as the default blockchain, but we do not have a connection for it in the connection pool. Did you pass right RPC configuration?")

    def check_default_chain_id(self):
        """Check that we are connected to the correct chain.

        The JSON-RPC node chain id should be the same as in the strategy module.
        """

        assert self.default_chain_id, "default_chain_id not set"
        web3 = self.get_default()
        assert web3.eth.chain_id == self.default_chain_id.value, f"Expected chain id {self.default_chain_id}, got {web3.eth.chain_id}"

    @classmethod
    def setup_from_environment(cls, gas_price_method: Optional[GasPriceMethod], **kwargs) -> "Web3Config":
        """Setup connections based on given RPC URLs.

        Read `JSON_RPC_BINANCE`, `JSON_RPC_POLYGON`, etc.
        environment variables.

        :param kwargs:
            {json_rpc_xxx: rpc URL} dict, as parsed by Typer.

        """

        web3config = Web3Config()

        web3config.gas_price_method = gas_price_method

        for chain_id in SUPPORTED_CHAINS:
            key = f"json_rpc_{chain_id.get_slug()}"

            rpc = kwargs.get(key)
            if rpc:
                redacted_url = get_url_domain(rpc)
                logger.info("Chain %s connects using %s", chain_id.name, redacted_url)
                web3config.connections[chain_id] = Web3Config.create_web3(rpc, gas_price_method)

        return web3config

    def has_any_connection(self):
        return len(self.connections) > 0
