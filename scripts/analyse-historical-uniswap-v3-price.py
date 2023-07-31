"""Check the Uniswap v3 price estimation at a historical block.

- Understand why slippage was what it was

- Check what was the estimated and executed sell WMATIC->USDC on Uniswap v3

- See the TX https://polygonscan.com/tx/0x5b76bf15bce4de5f5d6db8d929f13e28b11816f282ecd1522e4ec6eca3a1655e

"""
import os
from decimal import Decimal

from web3 import Web3, HTTPProvider

from eth_defi.token import fetch_erc20_details
from eth_defi.uniswap_v3.deployment import fetch_deployment
from eth_defi.uniswap_v3.price import get_onchain_price, estimate_sell_received_amount

params = {"path":"0x0d500b1d8e8ef31e21c99d1db9a6444d3adf12700001f42791bca1f2de4661ed88a30c99a7a9449aa84174","recipient":"0x19f61a2cdebccbf500b24a1330c46b15e5f54cbc","deadline":"9223372036854775808","amountIn":"14975601230579683413","amountOutMinimum":"10799953"}

amount_in = 14975601230579683413
path = params["path"]
# https://tradingstrategy.ai/trading-view/polygon/uniswap-v3/matic-usdc-fee-5
pool_address = "0xa374094527e1673a86de625aa59517c5de346d32"
block_estimated = 45_583_631
block_executed = 45_583_635

wmatic_address = "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270"
usdc_address = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
wmatic_amount = Decimal("14.975601230579683413")
fee_tier = 0.0005  # BPS

# What is the max slippage value for Uniswap,
# as slippage is irrelevant in our
# calculations
max_slippage = 10000

json_rpc_url = os.environ["JSON_RPC_POLYGON"]
web3 = Web3(HTTPProvider(json_rpc_url))

wmatic = fetch_erc20_details(web3, wmatic_address)
usdc = fetch_erc20_details(web3, usdc_address)

wmatic_amount_raw = wmatic.convert_to_raw(wmatic_amount)

mid_price_estimated = get_onchain_price(web3, pool_address, block_identifier=block_estimated)
mid_price_executed = get_onchain_price(web3, pool_address, block_identifier=block_executed)

print(f"Mid price when estimate at block {block_estimated:,}:", mid_price_estimated)
print(f"Mid price at the time of execution at block {block_executed:,}:", mid_price_executed)
print(f"Price difference {(mid_price_executed - mid_price_estimated) / mid_price_estimated * 100:.2f}%")

# Uniswap v4 deployment addresses are the same across the chains
# https://docs.uniswap.org/contracts/v3/reference/deployments
uniswap = fetch_deployment(
    web3,
    "0x1F98431c8aD98523631AE4a59f267346ea31F984",
    "0xE592427A0AEce92De3Edee1F18E0157C05861564",
    "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
    "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
)

estimated_sell_raw = estimate_sell_received_amount(
    uniswap,
    base_token_address=wmatic_address,
    quote_token_address=usdc_address,
    quantity=wmatic_amount_raw,
    target_pair_fee=int(fee_tier * 1_000_000),
    block_identifier=block_estimated,
    slippage=max_slippage,
)
estimated_sell = usdc.convert_to_decimals(estimated_sell_raw)

print(f"Estimated received quantity: {estimated_sell} USDC")

executed_sell_raw = estimate_sell_received_amount(
    uniswap,
    base_token_address=wmatic_address,
    quote_token_address=usdc_address,
    quantity=wmatic_amount_raw,
    target_pair_fee=int(fee_tier * 1_000_000),
    block_identifier=block_executed,
    slippage=max_slippage,
)
executed_sell = usdc.convert_to_decimals(executed_sell_raw)

print(f"Executed received quantity: {executed_sell} USDC")

print(f"Supposed price impact {(executed_sell - estimated_sell) / estimated_sell * 100:.2f}%")

