"""Web3 connection configuration."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from eth_defi.gas import GasPriceMethod, node_default_gas_price_strategy
from eth_defi.hotwallet import HotWallet
from eth_defi.middleware import http_retry_request_with_sleep_middleware
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.provider.broken_provider import set_block_tip_latency
from eth_defi.provider.multi_provider import (MultiProviderWeb3,
                                              create_multi_provider_web3)
from tradingstrategy.chain import ChainId
from web3 import HTTPProvider, Web3

from tradeexecutor.monkeypatch.web3 import \
    construct_sign_and_send_raw_middleware
from tradeexecutor.utils.url import get_url_domain

#: List of currently supportd EVM blockchains
SUPPORTED_CHAINS = [
    ChainId.ethereum,
    ChainId.avalanche,
    ChainId.polygon,
    ChainId.bsc,
    ChainId.arbitrum,
    ChainId.anvil,
    ChainId.base,
    ChainId.derive,
    ChainId.arbitrum_sepolia,
    ChainId.base_sepolia,
    ChainId.hyperliquid,
    ChainId.hyperliquid_testnet,
    ChainId.monad,
]

#: Override broken slugs in the tradingstrategy library.
#: ChainId.hyperliquid (999) reports "wanchain testnet" instead of "hyperliquid".
_CHAIN_SLUG_OVERRIDES: dict[ChainId, str] = {
    ChainId.hyperliquid: "hyperliquid",
    ChainId.hyperliquid_testnet: "hyperliquid_testnet",
}


def _get_chain_slug(chain_id: ChainId) -> str:
    """Get the RPC kwarg slug for a chain, with overrides for broken slugs."""
    return _CHAIN_SLUG_OVERRIDES.get(chain_id, chain_id.get_slug())

#: Funny chain ids used. e.g. with mainnet forks
TEST_CHAIN_IDS: List[ChainId] = [
    ChainId.ethereum_tester,
    ChainId.anvil,
]

logger = logging.getLogger(__name__)


def filter_rpc_kwargs_by_chain(chain_name: str, **rpc_kwargs) -> dict:
    """Filter JSON-RPC kwargs to keep only the selected chain.

    Used when multiple ``JSON_RPC_*`` environment variables are set but only
    one chain is needed (e.g. ``simulate`` mode).

    :param chain_name:
        Chain slug, e.g. ``"base"``, ``"arbitrum"``, ``"ethereum"``,
        ``"hyperliquid"``.
    :param rpc_kwargs:
        The ``json_rpc_*`` keyword arguments as passed to :func:`create_web3_config`.
    :return:
        Same dict with all non-matching RPC values set to ``None``.
    :raises AssertionError:
        If the slug is unknown or no RPC is configured for it.
    """
    # Try standard slug lookup first, then check overrides
    selected = ChainId.get_by_slug(chain_name)
    if selected is None:
        # Check if chain_name matches an override value
        for chain_id, slug in _CHAIN_SLUG_OVERRIDES.items():
            if slug == chain_name:
                selected = chain_id
                break
    assert selected is not None, f"Unknown chain slug: {chain_name}. Supported: {[_get_chain_slug(c) for c in SUPPORTED_CHAINS]}"
    target_key = f"json_rpc_{_get_chain_slug(selected)}"
    assert rpc_kwargs.get(target_key), (
        f"No JSON-RPC configured for chain {chain_name} "
        f"(expected env var JSON_RPC_{chain_name.upper()})"
    )
    return {k: (v if k == target_key else None) for k, v in rpc_kwargs.items()}


def collect_rpc_kwargs(
    *,
    json_rpc_binance=None,
    json_rpc_polygon=None,
    json_rpc_avalanche=None,
    json_rpc_ethereum=None,
    json_rpc_base=None,
    json_rpc_anvil=None,
    json_rpc_arbitrum=None,
    json_rpc_derive=None,
    json_rpc_arbitrum_sepolia=None,
    json_rpc_base_sepolia=None,
    json_rpc_hyperliquid=None,
    json_rpc_hyperliquid_testnet=None,
    json_rpc_monad=None,
    chain_name: str | None = None,
) -> dict:
    """Collect JSON-RPC kwargs and optionally filter to a single chain.

    Standardises the boilerplate of gathering Typer RPC params into a dict
    suitable for :func:`create_web3_config`.  When ``chain_name`` is given,
    delegates to :func:`filter_rpc_kwargs_by_chain` so only that chain's
    RPC survives.
    """
    rpc_kwargs = dict(
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_base=json_rpc_base,
        json_rpc_anvil=json_rpc_anvil,
        json_rpc_arbitrum=json_rpc_arbitrum,
        json_rpc_derive=json_rpc_derive,
        json_rpc_arbitrum_sepolia=json_rpc_arbitrum_sepolia,
        json_rpc_base_sepolia=json_rpc_base_sepolia,
        json_rpc_hyperliquid=json_rpc_hyperliquid,
        json_rpc_hyperliquid_testnet=json_rpc_hyperliquid_testnet,
        json_rpc_monad=json_rpc_monad,
    )
    if chain_name:
        rpc_kwargs = filter_rpc_kwargs_by_chain(chain_name, **rpc_kwargs)
    return rpc_kwargs


@dataclass
class Web3Config:
    """Advanced Web3 connection manager.

    Supports multiple blockchain connections.
    """

    #: Mapping of different connections for different chains
    connections: Dict[ChainId, MultiProviderWeb3] = field(default_factory=dict)

    #: How do we price our txs
    gas_price_method: Optional[GasPriceMethod] = None

    #: Chain id for single chain strategies
    default_chain_id: Optional[ChainId] = None

    #: Anvil forks per chain (for multi-chain simulation)
    anvils: dict = field(default_factory=dict)  # dict[ChainId, AnvilLaunch]

    #: Is this mainnet fork for simulation deployment
    mainnet_fork_simulation = False

    @property
    def anvil(self) -> Optional[AnvilLaunch]:
        """Backward-compatible single Anvil accessor."""
        if self.anvils:
            return next(iter(self.anvils.values()))
        return None

    @anvil.setter
    def anvil(self, value):
        """Backward-compatible setter."""
        if value is None:
            self.anvils.clear()
        else:
            self.anvils[ChainId.anvil] = value

    @staticmethod
    def create_web3(
        configuration_line: str,
        gas_price_method: Optional[GasPriceMethod] = None,
        unit_testing: bool=False,
        simulate: bool=False,
        mev_endpoint_disabled: bool=False,
        simulate_http_timeout: tuple[float, float] | None = None,
    ) -> MultiProviderWeb3:
        """Create a new Web3.py connection.

        :param configuration_line:
            JSON-RPC configuration line.

            May contain several space separated entries.

        :param gas_price_method:
            How do we estimate gas for a transaction
            If not given autodetect the method.

        :param unit_testing:
            Are we executing against unit testing JSON-RPC endpoints.

            If so set latency to zero.

        :param simulate:
            Set up Anvil mainnet fork for transaction simulation.

        :param mev_endpoint_disabled:
            MEV endpoints do not work when deploying contracts with Forge.

        """

        assert type(configuration_line) == str, f"Got: {configuration_line.__class__}"

        if simulate:
            # Use last given RPC for anvil,
            # because first one is likely MEV one
            last_rpc = configuration_line.split(" ")[-1]
            logger.info(f"Simulating transactions with Anvil, forking from {last_rpc}")
            anvil = launch_anvil(last_rpc, attempts=1)
            web3 = create_multi_provider_web3(
                anvil.json_rpc_url,
                switchover_noisiness=logging.TRADE,
                default_http_timeout=simulate_http_timeout,
            )
            web3.anvil = anvil
            web3.simulate = True
            configuration_line = anvil.json_rpc_url  # Override whatever configuration given earlier
        else:

            if mev_endpoint_disabled:
                configuration_items = configuration_line.split(" ")
                configuration_items = [c for c in configuration_items if not c.startswith("mev+https://")]
                configuration_line = " ".join(configuration_items)
                logger.info("mev_endpoint_disabled: running with RPCs %s", configuration_line)

            web3 = create_multi_provider_web3(configuration_line)

        # Read numeric chain id from JSON-RPC
        chain_id = web3.eth.chain_id

        if gas_price_method is None:
            if chain_id in (ChainId.ethereum.value, ChainId.ganache.value, ChainId.avalanche.value, ChainId.polygon.value, ChainId.anvil.value, ChainId.arbitrum.value, ChainId.base.value, ChainId.arbitrum_sepolia.value, ChainId.base_sepolia.value):
                # Ethereum supports maxBaseFee method (London hard fork)
                # Same for Avalanche C-chain https://twitter.com/avalancheavax/status/1389763933448323073

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

        try:
            chain_id_obj = ChainId(chain_id)
            chain_label = chain_id_obj.name
        except ValueError:
            # Unknown chain ID (e.g. HyperEVM testnet 998) — use raw int
            chain_label = str(chain_id)

        rpc_urls = [get_url_domain(rpc) for rpc in configuration_line.split()]
        logger.info("Chain %s connects using %s", chain_label, rpc_urls)

        logger.trade("Connected to chain: %s, gas pricing method: %s, providers %s",
                     chain_label,
                     gas_price_method.name,
                     rpc_urls,
                     )

        # London is the default method
        if gas_price_method == GasPriceMethod.legacy:
            logger.info("Setting up gas price middleware for Web3")
            web3.eth.set_gas_price_strategy(node_default_gas_price_strategy)

        if unit_testing:
            set_block_tip_latency(web3, 0)

        return web3

    def close(self, log_level: int | None = None):
        """Close all connections.

        :param level:
            Logging level to copy Anvil stdout
        """
        for anvil_instance in self.anvils.values():
            anvil_instance.close(log_level=log_level)
        self.anvils.clear()

    def has_chain_configured(self) -> bool:
        """Do we have one or more chains configured."""
        return len(self.connections) > 0

    def choose_single_chain(self, default_chain_id: Optional[ChainId] = None):
        """Set the default chain we are connected to.

        Ensure we have exactly 1 JSON-RPC endpoint configured.

        :param default_chain_id:
            Hint from the strategy module
        """
        if default_chain_id is not None:
            assert isinstance(default_chain_id, ChainId)
            self.set_default_chain(default_chain_id)
            self.check_default_chain_id()
        else:
            assert len(self.connections) == 1, f"Expected a single web3 connection, got JSON-RPC for chains {list(self.connections.keys())}"
            default_chain_id = next(iter(self.connections.keys()))
            self.set_default_chain(default_chain_id)
            self.check_default_chain_id()
        logger.info("Chosen to use a single blockchain. Chain id: %d: Provider: %s", default_chain_id, self.connections[default_chain_id].provider)

    def set_default_chain(self, chain_id: ChainId):
        """Set the default chain our strategy runs on.

        Most strategies are single chain strategies.
        Set the chain id we expect these strategies to run on.
        """
        assert isinstance(chain_id, ChainId), f"Attempt to set null chain as the default: {chain_id}"
        self.default_chain_id = chain_id

    def get_connection(self, chain_id: ChainId) -> Optional[Web3]:
        """Get a connection to a specific network.

        Hypercore (9999) shares the same RPC as Hyperliquid (999),
        so fall back transparently when no dedicated connection exists.
        """
        if chain_id not in self.connections and chain_id == ChainId.hypercore:
            return self.connections[ChainId.hyperliquid]
        return self.connections[chain_id]

    def get_default(self) -> Web3:
        """Getst the default connection.

        Assumes exactly 1 node connection available.
        """

        assert self.default_chain_id, "No default chain id set"
        try:
            return self.connections[self.default_chain_id]
        except KeyError:
            raise RuntimeError(f"We have {self.default_chain_id.name} configured as the default blockchain, but we do not have a connection for it in the connection pool. Did you pass right RPC configuration?")

    def check_default_chain_id(self):
        """Check that we are connected to the correct chain.

        The JSON-RPC node chain id should be the same as in the strategy module.
        """

        assert self.default_chain_id, "default_chain_id not set"
        web3 = self.get_default()

        if self.default_chain_id not in TEST_CHAIN_IDS:
            actual = web3.eth.chain_id
            expected = self.default_chain_id.value
            assert actual == expected, \
                f"Strategy expected chain id {self.default_chain_id} ({expected}), " \
                f"RPC says we got {actual}"

    @classmethod
    def setup_from_environment(
        cls,
       gas_price_method: Optional[GasPriceMethod],
       unit_testing: bool=False,
       simulate: bool=False,
       mev_endpoint_disabled: bool=False,
       simulate_http_timeout: tuple[float, float] | None = None,
       **kwargs
    ) -> "Web3Config":
        """Setup connections based on given RPC URLs.

        Read `JSON_RPC_BINANCE`, `JSON_RPC_POLYGON`, etc.
        environment variables.

        :param mev_endpoint_disabled:
            MEV endpoints do not work when deploying contracts with Forge.

        :param kwargs:
            {json_rpc_xxx: rpc URL} dict, as parsed by Typer.
        """

        web3config = Web3Config()

        web3config.gas_price_method = gas_price_method
        web3config.mainnet_fork_simulation = simulate

        # Lowercase all key names
        kwargs = {k.lower(): v for k, v in kwargs.items()}

        for chain_id in SUPPORTED_CHAINS:
            key = f"json_rpc_{_get_chain_slug(chain_id)}"
            configuration_line = kwargs.get(key)
            if configuration_line:
                web3config.connections[chain_id] = Web3Config.create_web3(
                    configuration_line,
                    gas_price_method,
                    unit_testing=unit_testing,
                    simulate=simulate,
                    mev_endpoint_disabled=mev_endpoint_disabled,
                    simulate_http_timeout=simulate_http_timeout,
                )

                if simulate:
                    web3config.anvils[chain_id] = getattr(web3config.connections[chain_id], "anvil")

        return web3config

    def has_any_connection(self):
        return len(self.connections) > 0

    def is_mainnet_fork(self) -> bool:
        """Is this connection a testing fork of a mainnet."""

        if self.mainnet_fork_simulation:
            return True

        if self.anvils:
            return True

        return ChainId.anvil in self.connections

    def add_hot_wallet_signing(self, hot_wallet: HotWallet):
        """Make web3.py native signing available in the console."""
        assert isinstance(hot_wallet, HotWallet)
        for web3 in self.connections.values():
            web3.middleware_onion.add(construct_sign_and_send_raw_middleware(hot_wallet.account))
            hot_wallet.sync_nonce(web3)
