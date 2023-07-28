"""Check the Uniswap v3 price estimation at a historical block."""
import os

from web3 import Web3, HTTPProvider

from eth_defi.uniswap_v3.price import get_onchain_price

params = {"path":"0x0d500b1d8e8ef31e21c99d1db9a6444d3adf12700001f42791bca1f2de4661ed88a30c99a7a9449aa84174","recipient":"0x19f61a2cdebccbf500b24a1330c46b15e5f54cbc","deadline":"9223372036854775808","amountIn":"14975601230579683413","amountOutMinimum":"10799953"}

amount_in = 14975601230579683413
path = params["path"]
# https://tradingstrategy.ai/trading-view/polygon/uniswap-v3/matic-usdc-fee-5
pool_address = "0xa374094527e1673a86de625aa59517c5de346d32"
block_estimated = 45_583_631
block_executed = 45_583_635

json_rpc_url = os.environ["JSON_RPC_POLYGON"]
web3 = Web3(HTTPProvider(json_rpc_url))

mid_price_estimated = get_onchain_price(web3, pool_address, block_identifier=block_estimated)
mid_price_executed = get_onchain_price(web3, pool_address, block_identifier=block_estimated)

print("Mid price when estimated:", mid_price_estimated)
print("Mid price at the time of execution:", mid_price_executed)


