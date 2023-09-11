"""Mock testing client that supports multiple exchanges' on-chain data to generate datasets.

TODO: refactor repeated code from UniswapV2MockClient

Needed to be in trade-executor due to working with UniswapV2TestingData and UniswapV3TestingData classes.
"""

import logging
from types import NoneType
from typing import Tuple, cast

from pyarrow import Table

from web3 import Web3, HTTPProvider
from eth_defi.abi import get_contract
from eth_defi.event_reader.conversion import decode_data, convert_uint256_bytes_to_address, convert_int256_bytes_to_int, convert_uint256_hex_string_to_address
from eth_defi.event_reader.filter import Filter
from eth_defi.event_reader.logresult import LogResult
from eth_defi.event_reader.reader import read_events
from eth_defi.token import fetch_erc20_details
from eth_typing import HexAddress, HexStr

from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType, ExchangeUniverse
from tradingstrategy.pair import DEXPair, PandasPairUniverse
from tradingstrategy.stablecoin import is_stablecoin_like
from tradingstrategy.testing.mock_client import MockClient

from tradeexecutor.testing.evm_uniswap_testing_data import UniswapV2TestData, UniswapV3TestData


logger = logging.getLogger(__name__)


class GenericMockClient(MockClient):
    """A mock client that reads data from a single Uniswap v2 exchange directly from the chain.

    Designed to run tests against test EVM backends where we cannot generate
    proper test data because of the backends being temporary. This way we can skip the ETL
    step and pretend that the data is just there, but still have meaningful interaction
    with trading strategies with pairs and trade execution and such.

    Currently supported

    - Exchanges

    - Pairs data

    Any data is read from the chain on construction, then cached for subsequent fetch calls.

    .. warning::

        This client is not suitable to iterate real on-chain data.
        Due to high amount of real pairs deployed, you will need to wait
        several hours for :py:methd:`read_onchain_data` to complete.

    """

    def __init__(
        self,
        web3: Web3,
        test_evm_uniswap_data: list[UniswapV2TestData | UniswapV3TestData],
        fee: float = 0.0030,
    ):

        self.test_evm_uniswap_data = test_evm_uniswap_data

        self.exchange_universe, self.pairs_table = GenericMockClient.read_onchain_data(
            web3,
            test_evm_uniswap_data,
            fee,
        )

        assert len(self.pairs_table) > 0, f"Could not read any pairs from on-chain data. Test_evm_uniswap_data: {test_evm_uniswap_data}."
    
    @staticmethod
    def get_item_for_all_test_data_members(test_evm_uniswap_data: list[UniswapV2TestData | UniswapV3TestData], item: str) -> list[str]:
        """Get the factory addresses from the test evm uniswap data.
        
        :param test_evm_uniswap_data:
            The test evm uniswap data to get the items from. Each element in the list is either an instance of UniswapV2TestData or UniswapV3TestData.
        
        :param item:
            The item to get from the test evm uniswap data. Should be one of "factory" or "exchange_slug" or "router" or "init_code_hash" or "exchange_id".

        :returns:
            The relevant items as a list of strings.
        """
        assert item in {"factory", "exchange_slug", "router", "init_code_hash", "exchange_id"}, f"item should be either factory or exchange_slug or router or init_code_hash or exchange_id, got {item}"
        items: list[str | None] = []
        for data_item in test_evm_uniswap_data:
            if not isinstance(data_item, UniswapV2TestData | UniswapV3TestData):
                raise NotImplementedError(f"data_item should be of type UniswapV2TestData or UniswapV3TestData, got {data_item}")

            if item in data_item.__dict__:
                items.append(data_item.__dict__[item])
            else:
                items.append(None)
        return items

    @staticmethod
    def read_onchain_data(
                web3: Web3,
                test_evm_uniswap_data: list[UniswapV2TestData | UniswapV3TestData],
                fee: float = 0.003,
    ) -> Tuple[ExchangeUniverse, Table]:
        """Reads Uniswap v2 data from EVM backend and creates tables for it.

        - Read data from a single Uniswap v2 compatible deployment

        - Read all PairCreated events and constructed Pandas DataFrame out of them

        :param fee:
            Uniswap v2 do not have fee information available on-chain, so we need to pass it.

            Default to 30 BPS.
        """

        chain_id = ChainId(web3.eth.chain_id)

        factory_addresses = GenericMockClient.get_item_for_all_test_data_members(test_evm_uniswap_data, "factory")
        exchange_slugs = GenericMockClient.get_item_for_all_test_data_members(test_evm_uniswap_data, "exchange_slug")
        router_addresses = GenericMockClient.get_item_for_all_test_data_members(test_evm_uniswap_data, "router")
        init_code_hashes = GenericMockClient.get_item_for_all_test_data_members(test_evm_uniswap_data, "init_code_hash")
        exchange_ids = GenericMockClient.get_item_for_all_test_data_members(test_evm_uniswap_data, "exchange_id")

        # Get contracts
        Factory_v2 = get_contract(web3, "sushi/UniswapV2Factory.json")
        Factory_v3 = get_contract(web3, 'uniswap_v3/UniswapV3Factory.json')

        start_block = 1
        end_block = web3.eth.block_number

        if isinstance(web3.provider, HTTPProvider):
            endpoint_uri = web3.provider.endpoint_uri
        else:
            endpoint_uri = str(web3.provider)

        # Assume logging is safe, because this mock client is only to be used with testing backends
        logger.info("Scanning PairCreated events, %d - %d, from %s, factories are %s",
                    start_block,
                    end_block,
                    endpoint_uri,
                    factory_addresses
                    )

        my_filter = Filter.create_filter(
            factory_addresses,
            [Factory_v2.events.PairCreated, Factory_v3.events.PoolCreated],
        )

        # Read through all the events, all the chain, using a single threaded slow loop.
        # Only suitable for test EVM backends.
        pairs = []
        log: LogResult
        hardcoded_pair_id = 1  # TODO: fix hardcoded pair id
        exchange_id_to_pair_count = {}
        for log in read_events(
            web3,
            start_block,
            end_block,
            filter=my_filter,
            extract_timestamps=None,
        ):
            # Signature this
            #
            #  event PairCreated(address indexed token0, address indexed token1, address pair, uint);
            #
            # topic 0 = keccak(event signature)
            # topic 1 = token 0
            # topic 2 = token 1
            # argument 0 = pair
            # argument 1 = pair id
            #
            # log for EthereumTester backend is
            #
            # {'type': 'mined',
            #  'logIndex': 0,
            #  'transactionIndex': 0,
            #  'transactionHash': HexBytes('0x2cf4563f8c275e5b5d7a4e5496bfbaf15cc00d530f15f730ac4a0decbc01d963'),
            #  'blockHash': HexBytes('0x7c0c6363bc8f4eac452a37e45248a720ff09f330117cdfac67640d31d140dc38'),
            #  'blockNumber': 6,
            #  'address': '0xF2E246BB76DF876Cef8b38ae84130F4F55De395b',
            #  'data': HexBytes('0x00000000000000000000000068931307edcb44c3389c507dab8d5d64d242e58f0000000000000000000000000000000000000000000000000000000000000001'),
            #  'topics': [HexBytes('0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9'),
            #   HexBytes('0x0000000000000000000000002946259e0334f33a064106302415ad3391bed384'),
            #   HexBytes('0x000000000000000000000000b9816fc57977d5a786e654c7cf76767be63b966e')],
            #  'context': None,
            #  'event': web3._utils.datatypes.PairCreated,
            #  'chunk_id': 1,
            #  'timestamp': None}
            #
            factory_address = log["address"]
            arguments = decode_data(log["data"])
            topics = log["topics"]
            token0 = convert_uint256_hex_string_to_address(topics[1])
            token1 = convert_uint256_hex_string_to_address(topics[2])

            if len(topics) == 3:
                # uniswap v2
                pair_address = convert_uint256_bytes_to_address(arguments[0])
                pair_id = convert_int256_bytes_to_int(arguments[1])
            elif len(topics) == 4:
                # uniswap v3
                pair_address = convert_uint256_bytes_to_address(arguments[1])

            token0_details = fetch_erc20_details(web3, token0)
            token1_details = fetch_erc20_details(web3, token1)

            # Our very primitive check to determine base token and quote token.
            # It's ok because this is a test backend
            if is_stablecoin_like(token0_details.symbol):
                quote_token_details = token0_details
                base_token_details = token1_details
            elif is_stablecoin_like(token1_details.symbol):
                quote_token_details = token1_details
                base_token_details = token0_details
            else:
                raise NotImplementedError(f"Does not know how to handle base-quote pairing for {token0_details} - {token1_details}.")

            exchange_slug = GenericMockClient.get_item_from_factory(factory_address, factory_addresses, exchange_slugs)
            exchange_id = GenericMockClient.get_item_from_factory(factory_address, factory_addresses, exchange_ids)
            
            # TODO: fix hardcoded pair id

            pair = DEXPair(
                    pair_id=hardcoded_pair_id,
                    chain_id=chain_id,
                    exchange_id=exchange_id,
                    exchange_slug=exchange_slug,
                    exchange_address=factory_address.lower(),
                    pair_slug=f"{base_token_details.symbol.lower()}-{quote_token_details.symbol.lower()}",
                    address=pair_address.lower(),
                    dex_type=GenericMockClient.get_exchange_type(exchange_slug),
                    base_token_symbol=base_token_details.symbol,
                    quote_token_symbol=quote_token_details.symbol,
                    token0_decimals=token0_details.decimals,
                    token1_decimals=token1_details.decimals,
                    token0_symbol=token0_details.symbol,
                    token1_symbol=token1_details.symbol,
                    token0_address=token0.lower(),
                    token1_address=token1.lower(),
                    buy_tax=0,
                    sell_tax=0,
                    transfer_tax=0,
                    fee=int(fee * 10_000),  #  Convert to BPS
                )
            hardcoded_pair_id += 1
            pairs.append(pair)
            exchange_id_to_pair_count[exchange_id] = exchange_id_to_pair_count.get(exchange_id, 0) + 1

        exchanges = []

        for i, router_address in enumerate(router_addresses):

            exchange_slug = exchange_slugs[i]
            exchange_id = exchange_ids[i]

            exchanges.append(Exchange(
                chain_id=chain_id,
                chain_slug=chain_id.get_slug(),
                exchange_slug=exchange_slug,
                exchange_id=exchange_id,
                address=factory_addresses[i].lower(),
                exchange_type=GenericMockClient.get_exchange_type(exchange_slug),
                pair_count=exchange_id_to_pair_count[exchange_id],
                default_router_address=router_address.lower(),
                init_code_hash=init_code_hashes[i],
            ))

        exchange_universe = ExchangeUniverse(exchanges={exchange.exchange_id: exchange for exchange in exchanges})

        pair_table = DEXPair.convert_to_pyarrow_table(pairs, check_schema=False)

        return exchange_universe, pair_table
    
    @staticmethod
    def get_exchange_type(exchange_slug: str) -> ExchangeType:
        """Get the exchange type from the exchange slug.
        
        :param exchange_slug:
            The exchange slug to get the exchange type from.

        :return:
            The exchange type.
        """

        if exchange_slug == "UniswapV2MockClient":
            return ExchangeType.uniswap_v2
        elif exchange_slug == "UniswapV3MockClient":
            return ExchangeType.uniswap_v3
        else:
            raise NotImplementedError(f"exchange_slug should be of type UniswapV2MockClient or UniswapV3MockClient, got {exchange_slug}")
        
    @staticmethod
    def get_item_from_factory(factory_address: str, factory_addresses: list[str], other: list[str]) -> str:
        """Get a pairs corresponding item from the factory address.
        
        :param factory_address:
            The factory address to get the exchange slug from.

        :return:
            The exchange slug.
        """

        assert len(factory_addresses) == len(other), f"factory_addresses and exchange_slugs should be of the same length, got {len(factory_addresses)} and {len(other)}"

        for i, factory in enumerate(factory_addresses):
            if factory_address.lower() == factory.lower():
                return other[i]
            
        raise NotImplementedError(f"factory_address should be in factory_addresses, got {factory_address} not in {factory_addresses}")





