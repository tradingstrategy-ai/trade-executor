"""Extract PancakeSwap information from on-chain.

Needed as an input for a trade executor.
"""
import os

from web3 import Web3, HTTPProvider

from eth_defi.uniswap_v2.deployment import fetch_deployment

rpc_url = os.environ["JSON_RPC_BINANCE"]

web3 = Web3(HTTPProvider(rpc_url, request_kwargs={"timeout": 2}))

pancakeswap_v2 = fetch_deployment(
    web3,
    "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73",
    "0x10ED43C718714eb63d5aA57B78B54704E256024E",
    # Taken from https://bscscan.com/address/0xca143ce32fe78f1f7019d7d551a6402fc5350c73#readContract
    init_code_hash="0x00fb7f630766e6a796048ea87d01acd3068e8ff67d078148a3fa3f4a84f69bd5",
    )


print(f"""
UNISWAP_V2_FACTORY_ADDRESS={pancakeswap_v2.factory.address}
UNISWAP_V2_ROUTER_ADDRESS={pancakeswap_v2.router.address}
UNISWAP_V2_INIT_CODE_HASH={pancakeswap_v2.init_code_hash}
""")