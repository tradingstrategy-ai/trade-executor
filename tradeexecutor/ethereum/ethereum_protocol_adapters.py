"""Default protocols support for EVM blockchains.

See :py:mod:`tradeexecutor.strategy.generic.pair_configurator`.
"""

import logging
from typing import Callable, Set

from web3 import Web3
from eth_defi.aave_v3.deployment import fetch_deployment as fetch_aave_deployment
from eth_defi.uniswap_v3.constants import UNISWAP_V3_DEPLOYMENTS
from eth_defi.uniswap_v3.deployment import fetch_deployment as fetch_uniswap_v3_deployment
from eth_defi.one_delta.deployment import fetch_deployment as fetch_1delta_deployment
from eth_defi.aave_v3.constants import AAVE_V3_DEPLOYMENTS
from eth_defi.one_delta.constants import ONE_DELTA_DEPLOYMENTS
from tradeexecutor.ethereum.routing_data import base_uniswap_v3_address_map
from tradeexecutor.ethereum.vault.vault_routing import VaultRouting

from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse, ExchangeType

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.generic.pair_configurator import PairConfigurator, ProtocolRoutingId, ProtocolRoutingConfig
from tradeexecutor.strategy.generic.default_protocols import default_match_router, default_supported_routers
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.pair import PandasPairUniverse

logger = logging.getLogger(__name__)


def get_exchange_type(
    exchange_universe: ExchangeUniverse,
    pair: TradingPairIdentifier,
) -> ExchangeType:
    assert pair.exchange_address is not None, f"Pair missing exchange_address: {pair}"
    exchange = exchange_universe.get_by_chain_and_factory(ChainId(pair.chain_id), pair.exchange_address)
    assert exchange is not None, f"Exchange address {pair.exchange_address} for pair {pair}: data not loaded"
    return exchange.exchange_type


def create_uniswap_v2_adapter(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
    routing_id: ProtocolRoutingId,
) -> ProtocolRoutingConfig:

    # TODO: Avoid circular imports for now
    from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
    from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_valuation import UniswapV2PoolRevaluator
    from tradeexecutor.ethereum.routing_data import create_uniswap_v2_compatible_routing

    logger.info("create_uniswap_v2_adapter(): %s", routing_id)

    assert routing_id.router_name == "uniswap-v2"
    assert len(strategy_universe.data_universe.chains) == 1
    assert len(strategy_universe.reserve_assets) == 1

    reserve = strategy_universe.get_reserve_asset()
    assert reserve.token_symbol in ("USDC", "USDT",)

    exchange_universe = strategy_universe.data_universe.exchange_universe
    chain_id = strategy_universe.get_single_chain()

    exchange = exchange_universe.get_by_chain_and_slug(chain_id, routing_id.exchange_slug)

    if exchange.exchange_slug == "quickswap":
        routing_model = create_uniswap_v2_compatible_routing(
            TradeRouting.quickswap_usdc,
            ReserveCurrency.usdc,
            chain_id,
        )
    elif exchange.exchange_slug == "sushi":
        routing_model = create_uniswap_v2_compatible_routing(
            TradeRouting.sushi_usdc,
            ReserveCurrency.usdc,
            chain_id,
        )
    elif exchange.exchange_slug == "pancakeswap-v2":
        routing_model = create_uniswap_v2_compatible_routing(
            TradeRouting.pancakeswap_usdt,
            ReserveCurrency.usdt,
            chain_id,
        )
    else:
        assert exchange.exchange_slug == "uniswap-v2", f"Expected uniswap-v2, got exchange slug {exchange.exchange_slug}"
        routing_model = create_uniswap_v2_compatible_routing(
            TradeRouting.uniswap_v2_usdc,
            ReserveCurrency.usdc,
            chain_id,
        )

    pricing_model = UniswapV2LivePricing(
        web3,
        strategy_universe.data_universe.pairs,
        routing_model,
    )

    valuation_model = UniswapV2PoolRevaluator(pricing_model)

    return ProtocolRoutingConfig(
        routing_id=routing_id,
        routing_model=routing_model,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
    )


def create_uniswap_v3_adapter(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
    routing_id: ProtocolRoutingId,
    chain_id_override: ChainId | None = None,
) -> ProtocolRoutingConfig:
    """Always the same."""

    # TODO: Avoid circular imports for now
    from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import UniswapV3LivePricing
    from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3Routing
    from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_valuation import UniswapV3PoolRevaluator
    from tradeexecutor.ethereum.routing_data import uniswap_v3_address_map

    assert routing_id.router_name == "uniswap-v3"
    assert len(strategy_universe.reserve_assets) == 1

    if chain_id_override is not None:
        chain_id = chain_id_override
        # For satellite chains, find the quote token used on that chain
        # (e.g. Base USDC for Uniswap v3 on Base).
        # Match pairs via exchange_id since exchange_slug may not be set on DEXPair.
        exchange_ids = set()
        for xc in strategy_universe.data_universe.exchange_universe.exchanges.values():
            if xc.exchange_slug == routing_id.exchange_slug:
                exchange_ids.add(xc.exchange_id)
        reserve_asset = None
        for dex_pair in strategy_universe.data_universe.pairs.iterate_pairs():
            if dex_pair.chain_id == chain_id and dex_pair.exchange_id in exchange_ids:
                reserve_asset = AssetIdentifier(
                    chain_id=chain_id.value,
                    address=dex_pair.quote_token_address,
                    token_symbol=dex_pair.quote_token_symbol,
                    decimals=dex_pair.quote_token_decimals,
                )
                break
        if reserve_asset is None:
            reserve_asset = strategy_universe.get_reserve_asset()
    else:
        chain_id = strategy_universe.get_single_chain()
        reserve_asset = strategy_universe.get_reserve_asset()

    allowed_intermediary_pairs = UNISWAP_V3_INTERMEDIATE.get(chain_id, {})

    if chain_id == ChainId.base:
        # Special case for Base chain
        address_map = UNISWAP_V3_DEPLOYMENTS["base"]
    elif chain_id == ChainId.binance:
        address_map = UNISWAP_V3_DEPLOYMENTS["binance"]
    elif chain_id in (ChainId.ethereum, ChainId.polygon, ChainId.arbitrum):
        address_map = uniswap_v3_address_map
    else:
        raise NotImplementedError(f"Chain {chain_id} not supported for Uniswap v3 - check address maps")

    # TODO: Add intermediate tokens
    routing_model = UniswapV3Routing(
        address_map=address_map,
        chain_id=chain_id,
        reserve_token_address=reserve_asset.address,
        allowed_intermediary_pairs=allowed_intermediary_pairs,
    )

    pricing_model = UniswapV3LivePricing(
        web3,
        strategy_universe.data_universe.pairs,
        routing_model,
    )

    valuation_model = UniswapV3PoolRevaluator(pricing_model)

    return ProtocolRoutingConfig(
        routing_id=routing_id,
        routing_model=routing_model,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
    )


def create_1delta_adapter(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
    routing_id: ProtocolRoutingId,
) -> ProtocolRoutingConfig:

    # TODO: Avoid circular imports for now
    from tradeexecutor.ethereum.one_delta.one_delta_live_pricing import OneDeltaLivePricing
    from tradeexecutor.ethereum.one_delta.one_delta_routing import OneDeltaRouting
    from tradeexecutor.ethereum.one_delta.one_delta_valuation import OneDeltaPoolRevaluator


    assert routing_id.router_name == "1delta"
    assert routing_id.lending_protocol_slug == "aave"
    assert routing_id.exchange_slug == "uniswap-v3"

    assert len(strategy_universe.data_universe.chains) == 1
    assert len(strategy_universe.reserve_assets) == 1
    chain_id = strategy_universe.get_single_chain()
    chain_slug = chain_id.get_slug()
    reserve_asset = strategy_universe.get_reserve_asset()

    assert chain_slug in AAVE_V3_DEPLOYMENTS, f"Chain {chain_slug} not supported for Aave v3"
    assert chain_slug in ONE_DELTA_DEPLOYMENTS, f"Chain {chain_slug} not supported for 1delta"

    uniswap_v3_deployment = fetch_uniswap_v3_deployment(
        web3,
        "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
        "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
    )

    aave_v3_deployment = fetch_aave_deployment(
        web3,
        pool_address=AAVE_V3_DEPLOYMENTS[chain_slug]["pool"],
        data_provider_address=AAVE_V3_DEPLOYMENTS[chain_slug]["data_provider"],
        oracle_address=AAVE_V3_DEPLOYMENTS[chain_slug]["oracle"],
    )

    one_delta_deployment = fetch_1delta_deployment(
        web3,
        flash_aggregator_address=ONE_DELTA_DEPLOYMENTS[chain_slug]["broker_proxy"],
        broker_proxy_address=ONE_DELTA_DEPLOYMENTS[chain_slug]["broker_proxy"],
        quoter_address=ONE_DELTA_DEPLOYMENTS[chain_slug]["quoter"],
    )

    address_map = {
        "one_delta_broker_proxy": one_delta_deployment.broker_proxy.address,
        "one_delta_quoter": one_delta_deployment.quoter.address,
        "aave_v3_pool": aave_v3_deployment.pool.address,
        "aave_v3_data_provider": aave_v3_deployment.data_provider.address,
        "aave_v3_oracle": aave_v3_deployment.oracle.address,
        "factory": uniswap_v3_deployment.factory.address,
        "router": uniswap_v3_deployment.swap_router.address,
        "position_manager": uniswap_v3_deployment.position_manager.address,
        "quoter": uniswap_v3_deployment.quoter.address
    }

    # TODO: Add intermediate tokens
    routing_model = OneDeltaRouting(
        address_map=address_map,
        chain_id=chain_id,
        reserve_token_address=reserve_asset.address,
        allowed_intermediary_pairs={},
    )

    pricing_model = OneDeltaLivePricing(
        web3,
        strategy_universe.data_universe.pairs,
        routing_model,
    )

    valuation_model = OneDeltaPoolRevaluator(pricing_model)

    return ProtocolRoutingConfig(
        routing_id=routing_id,
        routing_model=routing_model,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
    )


def create_aave_v3_adapter(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
    routing_id: ProtocolRoutingId,
) -> ProtocolRoutingConfig:

    # TODO: Avoid circular imports for now
    from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import UniswapV3LivePricing
    from tradeexecutor.ethereum.eth_valuation import EthereumPoolRevaluator
    from tradeexecutor.ethereum.aave_v3.aave_v3_routing import AaveV3Routing

    assert routing_id.router_name == "aave-v3"
    assert routing_id.lending_protocol_slug == "aave_v3"

    assert len(strategy_universe.data_universe.chains) == 1
    assert len(strategy_universe.reserve_assets) == 1
    chain_id = strategy_universe.get_single_chain()
    chain_slug = chain_id.get_slug()
    reserve_asset = strategy_universe.get_reserve_asset()

    assert chain_slug in AAVE_V3_DEPLOYMENTS, f"Chain {chain_slug} not supported for Aave v3"

    aave_v3_deployment = fetch_aave_deployment(
        web3,
        pool_address=AAVE_V3_DEPLOYMENTS[chain_slug]["pool"],
        data_provider_address=AAVE_V3_DEPLOYMENTS[chain_slug]["data_provider"],
        oracle_address=AAVE_V3_DEPLOYMENTS[chain_slug]["oracle"],
        ausdc_address=AAVE_V3_DEPLOYMENTS[chain_slug].get("ausdc"),
    )

    address_map = {
        "aave_v3_pool": aave_v3_deployment.pool.address,
        "aave_v3_data_provider": aave_v3_deployment.data_provider.address,
        "aave_v3_oracle": aave_v3_deployment.oracle.address,
    }

    routing_model = AaveV3Routing(
        address_map=address_map,
        chain_id=chain_id,
        reserve_token_address=reserve_asset.address,
        allowed_intermediary_pairs={},
    )

    pricing_model = UniswapV3LivePricing(
        web3,
        strategy_universe.data_universe.pairs,
        routing_model,
    )

    valuation_model = EthereumPoolRevaluator(pricing_model)

    return ProtocolRoutingConfig(
        routing_id=routing_id,
        routing_model=routing_model,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
    )



def create_vault_adapter(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
    routing_id: ProtocolRoutingId,
) -> ProtocolRoutingConfig:

    # TODO: Avoid circular imports for now
    from tradeexecutor.ethereum.vault.vault_live_pricing import VaultPricing
    from tradeexecutor.ethereum.vault.vault_valuation import VaultValuator
    # from tradeexecutor.ethereum.vault.vault_routing import VaultRouting

    logger.info("create_vault_adapter(): %s", routing_id)

    assert routing_id.router_name == "vault"
    assert len(strategy_universe.data_universe.chains) == 1
    assert len(strategy_universe.reserve_assets) == 1

    reserve = strategy_universe.get_reserve_asset()
    assert reserve.token_symbol in ("USDC", "USDT",)
    chain_id = strategy_universe.get_single_chain()

    routing_model = VaultRouting(reserve.address)
    pricing_model = VaultPricing(web3)
    valuation_model = VaultValuator(pricing_model)

    return ProtocolRoutingConfig(
        routing_id=routing_id,
        routing_model=routing_model,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
    )


def create_freqtrade_adapter(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
    routing_id: ProtocolRoutingId,
) -> ProtocolRoutingConfig:
    """Create Freqtrade routing adapter.

    Builds FreqtradeConfig for each Freqtrade pair by combining pair.other_data
    with environment variables for sensitive credentials.

    Environment variable naming: FREQTRADE_{FREQTRADE_ID}_USERNAME and _PASSWORD
    """

    # TODO: Avoid circular imports for now
    from tradeexecutor.strategy.freqtrade.freqtrade_routing import FreqtradeRoutingModel
    from tradeexecutor.strategy.freqtrade.freqtrade_pricing import FreqtradePricingModel
    from tradeexecutor.strategy.freqtrade.freqtrade_valuation import FreqtradeValuator
    from tradeexecutor.strategy.freqtrade.config import (
        FreqtradeConfig,
        OnChainTransferExchangeConfig,
        AsterExchangeConfig,
        HyperliquidExchangeConfig,
        OrderlyExchangeConfig,
    )
    import os
    from decimal import Decimal

    logger.info("create_freqtrade_adapter(): %s", routing_id)

    assert routing_id.router_name == "freqtrade"
    assert len(strategy_universe.reserve_assets) == 1

    reserve = strategy_universe.get_reserve_asset()
    assert reserve.token_symbol in ("USDC", "USDT",)

    # Collect all Freqtrade pairs and build configs
    freqtrade_configs = {}

    if strategy_universe.data_universe.pairs:
        for pair in strategy_universe.data_universe.pairs.iterate_pairs():
            if not pair.is_freqtrade():
                continue

            freqtrade_id = pair.other_data.get("freqtrade_id")
            if not freqtrade_id:
                raise ValueError(f"Freqtrade pair {pair} missing freqtrade_id in other_data")

            # Skip if already configured (same bot used by multiple pairs)
            if freqtrade_id in freqtrade_configs:
                continue

            # Get credentials from environment variables
            username_key = f"FREQTRADE_{freqtrade_id.upper().replace('-', '_')}_USERNAME"
            password_key = f"FREQTRADE_{freqtrade_id.upper().replace('-', '_')}_PASSWORD"

            api_username = os.environ.get(username_key)
            api_password = os.environ.get(password_key)

            if not api_username or not api_password:
                raise ValueError(
                    f"Missing Freqtrade credentials for {freqtrade_id}. "
                    f"Set {username_key} and {password_key} environment variables"
                )

            # Build exchange config from other_data
            transfer_method = pair.other_data.get("freqtrade_transfer_method")
            exchange_config = None

            if transfer_method == "on_chain_transfer":
                recipient_address = pair.other_data.get("freqtrade_recipient_address")
                if not recipient_address:
                    raise ValueError(f"on_chain_transfer requires freqtrade_recipient_address for {freqtrade_id}")
                exchange_config = OnChainTransferExchangeConfig(
                    recipient_address=recipient_address,
                    fee_tolerance=Decimal(pair.other_data.get("freqtrade_fee_tolerance", "1.0")),
                    confirmation_timeout=pair.other_data.get("freqtrade_confirmation_timeout", 600),
                    poll_interval=pair.other_data.get("freqtrade_poll_interval", 10),
                )
            elif transfer_method == "aster":
                vault_address = pair.other_data.get("freqtrade_vault_address")
                if not vault_address:
                    raise ValueError(f"aster requires freqtrade_vault_address for {freqtrade_id}")
                exchange_config = AsterExchangeConfig(
                    vault_address=vault_address,
                    broker_id=pair.other_data.get("freqtrade_broker_id", 0),
                    fee_tolerance=Decimal(pair.other_data.get("freqtrade_fee_tolerance", "1.0")),
                    confirmation_timeout=pair.other_data.get("freqtrade_confirmation_timeout", 600),
                    poll_interval=pair.other_data.get("freqtrade_poll_interval", 10),
                )
            elif transfer_method == "hyperliquid":
                vault_address = pair.other_data.get("freqtrade_vault_address")
                if not vault_address:
                    raise ValueError(f"hyperliquid requires freqtrade_vault_address for {freqtrade_id}")
                exchange_config = HyperliquidExchangeConfig(
                    vault_address=vault_address,
                    is_mainnet=pair.other_data.get("freqtrade_is_mainnet", True),
                    fee_tolerance=Decimal(pair.other_data.get("freqtrade_fee_tolerance", "1.0")),
                    confirmation_timeout=pair.other_data.get("freqtrade_confirmation_timeout", 600),
                    poll_interval=pair.other_data.get("freqtrade_poll_interval", 10),
                )
            elif transfer_method == "orderly_vault":
                vault_address = pair.other_data.get("freqtrade_vault_address")
                orderly_account_id = pair.other_data.get("freqtrade_orderly_account_id")
                broker_id = pair.other_data.get("freqtrade_broker_id")
                if not vault_address or not orderly_account_id or not broker_id:
                    raise ValueError(
                        f"orderly_vault requires freqtrade_vault_address, "
                        f"freqtrade_orderly_account_id, and freqtrade_broker_id for {freqtrade_id}"
                    )
                exchange_config = OrderlyExchangeConfig(
                    vault_address=vault_address,
                    orderly_account_id=orderly_account_id,
                    broker_id=broker_id,
                    token_id=pair.other_data.get("freqtrade_token_id"),
                    fee_tolerance=Decimal(pair.other_data.get("freqtrade_fee_tolerance", "1.0")),
                    confirmation_timeout=pair.other_data.get("freqtrade_confirmation_timeout", 600),
                    poll_interval=pair.other_data.get("freqtrade_poll_interval", 10),
                )

            # Build FreqtradeConfig
            config = FreqtradeConfig(
                freqtrade_id=freqtrade_id,
                api_url=pair.other_data["freqtrade_api_url"],
                api_username=api_username,
                api_password=api_password,
                exchange_name=pair.other_data["freqtrade_exchange"],
                reserve_currency=reserve.address,
                exchange=exchange_config,
            )

            freqtrade_configs[freqtrade_id] = config

    if not freqtrade_configs:
        raise ValueError("No Freqtrade pairs found in universe")

    # Create routing, pricing, and valuation models
    routing_model = FreqtradeRoutingModel(freqtrade_configs)

    # Create FreqtradeClients for pricing model
    from tradeexecutor.strategy.freqtrade.freqtrade_client import FreqtradeClient
    freqtrade_clients = {}
    for freqtrade_id, config in freqtrade_configs.items():
        freqtrade_clients[freqtrade_id] = FreqtradeClient(
            config.api_url,
            config.api_username,
            config.api_password,
        )

    pricing_model = FreqtradePricingModel(freqtrade_clients)
    valuation_model = FreqtradeValuator(pricing_model)

    return ProtocolRoutingConfig(
        routing_id=routing_id,
        routing_model=routing_model,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
    )


def create_exchange_account_adapter(
    routing_id: ProtocolRoutingId,
    account_value_func: Callable | None = None,
) -> ProtocolRoutingConfig:
    """Create adapter for exchange account pairs (Derive, etc.).

    Exchange account positions are valued via an external API,
    not through on-chain routing.
    """
    from tradeexecutor.exchange_account.pricing import ExchangeAccountPricingModel
    from tradeexecutor.exchange_account.valuation import ExchangeAccountValuator

    assert account_value_func is not None, \
        "account_value_func is required for exchange account routing. " \
        "Set DERIVE_SESSION_PRIVATE_KEY or similar credentials."

    pricing_model = ExchangeAccountPricingModel(account_value_func)
    valuation_model = ExchangeAccountValuator(pricing_model)

    return ProtocolRoutingConfig(
        routing_id=routing_id,
        routing_model=None,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
    )


def create_cctp_bridge_adapter(
    web3config: "Web3Config",
    routing_id: ProtocolRoutingId,
) -> ProtocolRoutingConfig:
    """Create adapter for CCTP bridge pairs.

    CCTP bridge positions transfer USDC between chains via Circle's
    Cross-Chain Transfer Protocol. Price is always 1:1 with zero fees.

    :param web3config:
        Web3Config with connections to all chains involved in bridging.
    :param routing_id:
        The protocol routing identifier.
    """
    from tradeexecutor.ethereum.cctp.pricing import CctpBridgePricingModel
    from tradeexecutor.ethereum.cctp.routing import CctpBridgeRouting
    from tradeexecutor.ethereum.cctp.valuation import CctpBridgeValuationModel

    routing_model = CctpBridgeRouting(web3config)
    pricing_model = CctpBridgePricingModel()
    valuation_model = CctpBridgeValuationModel()

    return ProtocolRoutingConfig(
        routing_id=routing_id,
        routing_model=routing_model,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
    )


class EthereumPairConfigurator(PairConfigurator):
    """Set up routes for EVM trading pairs.

    Supported protocols

    - 1delta

    - Uniswap v2 likes

    - Uniswap v3 likes

    - Aave v3

    - ERC-4626 vaults

    - CCTP bridge (cross-chain USDC)
    """

    def __init__(
        self,
        web3: Web3,
        strategy_universe: TradingStrategyUniverse | None,
        account_value_func: Callable | None = None,
        web3config: "Web3Config | None" = None,
    ):
        """Initialise pair configuration.

        :param web3:
            Web3 connection to the active blockchain (default chain).

        :param strategy_universe:
            The initial strategy universe.

            TODO: Currently only reserve currency, exchange and pair data is used.
            Candle data is discarded.

        :param account_value_func:
            Optional function to query exchange account values.
            Required when the universe contains exchange account pairs.
            Signature: (pair: TradingPairIdentifier) -> Decimal

        :param web3config:
            Optional Web3Config with connections to multiple chains.
            Required for CCTP bridge routing which needs access to both
            source and destination chain connections.
        """

        assert isinstance(web3, Web3)
        assert isinstance(strategy_universe, TradingStrategyUniverse)

        self.web3 = web3
        self.web3config = web3config
        self.account_value_func = account_value_func

        super().__init__(strategy_universe)

    def get_web3_for_chain(self, chain_id: int) -> Web3:
        """Get a Web3 connection for a specific chain.

        Falls back to self.web3 if no Web3Config is available.

        :param chain_id:
            The chain id to get a connection for.
        :return:
            Web3 connection for the specified chain.
        """
        if self.web3config is not None:
            connection = self.web3config.get_connection(ChainId(chain_id))
            if connection is not None:
                return connection
        return self.web3

    def get_supported_routers(self) -> Set[ProtocolRoutingId]:
        return default_supported_routers(self.strategy_universe)

    def create_config(self, routing_id: ProtocolRoutingId, three_leg_resolution=True, pairs: PandasPairUniverse=None) -> ProtocolRoutingConfig:
        if routing_id.router_name == "1delta":
            return create_1delta_adapter(self.web3, self.strategy_universe, routing_id)
        elif routing_id.router_name == "uniswap-v2":
            return create_uniswap_v2_adapter(self.web3, self.strategy_universe, routing_id)
        elif routing_id.router_name == "uniswap-v3":
            # For multichain universes, determine the chain from the exchange's pairs
            chain_id_override = None
            if len(self.strategy_universe.data_universe.chains) > 1 and routing_id.exchange_slug:
                for xc in self.strategy_universe.data_universe.exchange_universe.exchanges.values():
                    if xc.exchange_slug == routing_id.exchange_slug:
                        chain_id_override = xc.chain_id
                        break
            web3 = self.get_web3_for_chain(chain_id_override.value) if chain_id_override else self.web3
            return create_uniswap_v3_adapter(web3, self.strategy_universe, routing_id, chain_id_override=chain_id_override)
        elif routing_id.router_name == "aave-v3":
            return create_aave_v3_adapter(self.web3, self.strategy_universe, routing_id)
        elif routing_id.router_name == "vault":
            return create_vault_adapter(self.web3, self.strategy_universe, routing_id)
        elif routing_id.router_name == "freqtrade":
            return create_freqtrade_adapter(self.web3, self.strategy_universe, routing_id)
        elif routing_id.router_name == "exchange_account":
            return create_exchange_account_adapter(routing_id, self.account_value_func)
        elif routing_id.router_name == "cctp-bridge":
            return create_cctp_bridge_adapter(self.web3config, routing_id)
        else:
            raise NotImplementedError(f"Cannot route exchange {routing_id}")

    def match_router(self, pair: TradingPairIdentifier) -> ProtocolRoutingId:
        return default_match_router(self.strategy_universe, pair)


UNISWAP_V3_INTERMEDIATE = {
    ChainId.ethereum: {
        # Route WETH through WETH-USDC 5 BPS
        # https://tradingstrategy.ai/trading-view/ethereum/uniswap-v3/eth-usdc-fee-5
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
    },

    ChainId.base: {
        # Route WETH through WETH-USDC 5 BPS
        # https://coinmarketcap.com/dexscan/base/0xd0b53d9277642d899df5c87a3966a349a798f224/
        "0x4200000000000000000000000000000000000006": "0xd0b53d9277642d899df5c87a3966a349a798f224",
    }
}
