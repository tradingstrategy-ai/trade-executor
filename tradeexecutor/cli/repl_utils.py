"""REPL helper commands for CLI console."""
import datetime
from decimal import Decimal
from pprint import pformat

from eth_typing import HexAddress

from web3 import Web3
from web3.contract.contract import ContractFunction

from tqdm_loggable.auto import tqdm
from tabulate import tabulate

from eth_defi.confirmation import broadcast_and_wait_transactions_to_complete
from eth_defi.cow.quote import fetch_quote
from eth_defi.erc_4626.classification import create_vault_instance, create_vault_instance_autodetect
from eth_defi.erc_4626.core import get_vault_protocol_name, ERC4626Feature
from eth_defi.gas import estimate_gas_price, apply_gas
from eth_defi.hotwallet import HotWallet, SignedTransactionWithNonce
from eth_defi.lagoon.cowswap import approve_cow_swap, presign_and_broadcast, execute_presigned_cowswap_order
from eth_defi.lagoon.vault import LagoonVault
from eth_defi.token import fetch_erc20_details
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.token_cache import get_default_token_cache
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def _broadcast_tx(
    hot_wallet: HotWallet,
    bound_func: ContractFunction,
    value: int | None = None,
    tx_params: dict | None = None,
    defautl_gas_limit: int = 1_000_000,
) -> SignedTransactionWithNonce:
    """Craft a transaction payload to a smart contract function and broadcast it from our hot wallet.

    :param value:
        ETH attached to the transaction
    """
    global _tx_count

    _tx_count += 1

    # Create signed transactions from Web3.py contract calls
    # and use our broadcast waiter function to send out these txs onchain
    web3 = bound_func.w3
    gas_price_suggestion = estimate_gas_price(web3)
    tx_params = apply_gas(tx_params or {}, gas_price_suggestion)

    if not "gas" in tx_params:
        # Use default gas limit if not specified,
        # don't try to estimate
        tx_params["gas"] = defautl_gas_limit

    tx = hot_wallet.sign_bound_call_with_new_nonce(bound_func, value=value, tx_params=tx_params)
    print(f"Broadcasting tx #{_tx_count}: {tx.hash.hex()}, calling {bound_func.fn_name or '<unknown>'}() with account nonce {tx.nonce}")
    # Raises if the tx reverts
    broadcast_and_wait_transactions_to_complete(
        web3,
        [tx],
    )
    return tx


def list_vaults(
    console_context: dict,
    token_cache: "TokenDiskCache | None" = None,
):
    """List all vaults in the strategy universe, table formatted.

    Example:

    .. code-block:: python

        from tradeexecutor.cli.repl_utils import list_vaults
        list_vaults(locals())

    :param token_cache:
        Optional token cache. If not provided, uses default.
    """
    web3 = console_context["web3"]
    strategy_universe: TradingStrategyUniverse = console_context["strategy_universe"]
    vault_pairs = [pair for pair in strategy_universe.iterate_pairs() if pair.is_vault()]

    # Prepare vault instances to whitelist, and check
    # we can raad onchain data of them and there are not broken vaults/addresses

    if token_cache is None:
        token_cache = get_default_token_cache()

    vaults = []
    data = []
    broken_vaults = []
    for pair in tqdm(vault_pairs, desc="Processing vaults"):
        addr = pair.pool_address
        features = pair.get_vault_features()
        vault = create_vault_instance(
            web3,
            addr,
            features=features,
            token_cache=token_cache,
        )
        protocol_name = get_vault_protocol_name(vault.features)
        vaults.append(vault)
        data.append(
            {
                "Name": vault.name,
                "Address": addr,
                "Denomination": vault.denomination_token.symbol,
                "Protocol": protocol_name,
                "Features": ", ".join(f.value for f in vault.features),
            }
        )

        if ERC4626Feature.broken in vault.features:
            broken_vaults.append(vault)

    data = sorted(data, key=lambda x: x["Name"])

    # Display what we are about to whitelist
    table_fmt = tabulate(
        data,
        headers="keys",
        tablefmt="fancy_grid",
    )
    print("The following vaults are in the strategy universe:")
    print(table_fmt)



def deposit_4626(
    console_context: dict,
    address: HexAddress | str,
    amount_usd: float,
):
    """Make a command line trade to deposit into an ERC-4626 vault.

    Example:

    .. code-block:: python

        from tradeexecutor.cli.repl_utils import deposit_4626
        deposit_4626(
            locals(),
            # Plutus hedge token
            "0x58bfc95a864e18e8f3041d2fcd3418f48393fe6a",
            100.00,
        )

        # USDn2
        # 0x4a3f7dd63077cde8d7eff3c958eb69a3dd7d31a9
        deposit_4626(
            locals(),
            "0x4a3f7dd63077cde8d7eff3c958eb69a3dd7d31a9",
            100.00,
        )

        # Umami
        from tradeexecutor.cli.repl_utils import deposit_4626
        deposit_4626(
            locals(),
            "0x959f3807f0aa7921e18c78b00b2819ba91e52fef",
            100.00,
        )

        # Gains
        from tradeexecutor.cli.repl_utils import deposit_4626
        deposit_4626(
            locals(),
            "0xd3443ee1e91af28e5fb858fbd0d72a63ba8046e0",
            100.00,
        )

        # thBill
        #
        from tradeexecutor.cli.repl_utils import deposit_4626
        deposit_4626(
            locals(),
            "0x64ca76e2525fc6ab2179300c15e343d73e42f958",
            100.00,
        )

    """

    strategy_universe: TradingStrategyUniverse = console_context["strategy_universe"]
    pricing_model: PricingModel = console_context["pricing_model"]
    execution_model: EthereumExecution = console_context["execution_model"]
    routing_model: EthereumExecution = console_context["routing_model"]
    our_vault: LagoonVault = console_context["vault"]
    state: State = console_context["state"]
    web3: Web3 = console_context["web3"]
    store: JSONFileStore = console_context["store"]

    pair = strategy_universe.get_pair_by_smart_contract(address)
    vault = create_vault_instance_autodetect(
        web3,
        pair.pool_address,
    )
    web3 = console_context["web3"]

    position_manager = PositionManager(
        datetime.datetime.utcnow(),
        universe=strategy_universe,
        state=state,
        pricing_model=pricing_model,
        default_slippage_tolerance=0.20,
    )

    # Do guard checks
    trading_strategy_module = our_vault.trading_strategy_module
    trading_strategy_module_version = trading_strategy_module.functions.getTradingStrategyModuleVersion().call()

    print(f"Our vault is {our_vault.name} ({our_vault.fetch_version().value}) with TradingStrategyModule at {trading_strategy_module.address} ({trading_strategy_module_version})")

    print(f"About to deposit {amount_usd} USD to vault at {vault.name} ({pair.base.token_symbol} / {pair.quote.token_symbol})")
    print("Proceed [y/N]? ", end="")
    answer = input().strip().lower()
    if answer != "y":
        print("Aborting")
        return

    # Deposit to the vault
    trades = position_manager.open_spot(
        pair,
        value=amount_usd,
    )
    t = trades[0]

    routing_state_details = execution_model.get_routing_state_details()
    routing_state = routing_model.create_routing_state(strategy_universe, routing_state_details)

    execution_model.initialize()

    execution_model.execute_trades(
        datetime.datetime.utcnow(),
        state,
        trades,
        routing_model,
        routing_state,
        check_balances=True,
    )

    assert t.is_success(), f"Trade failed: {t.get_revert_reason()}"

    store.sync(state)


def swap_cow_interactive(
    console_context: dict,
    in_token: HexAddress | str,
    out_token: HexAddress | str,
    amount_in: float,
    max_slippage = 0.01,
):
    """Swap some tokens (stablecoins) using CowSwap.

    - Get a quote from CowSwap
    - Ask yes/no confirmation

    Example:

    .. code-block:: python

        # from tradeexecutor.cli.repl_utils import swap_cow_interactive

        # Swap from USDC to crvUSD on Arbitrum
        usdc = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
        crvusd = "0x498Bf2B1e120FeD3ad3D42EA2165E9b73f99C1e5"

        swap_cow_interactive(
            locals(),
            in_token=usdc,
            out_token=crvusd,
            amount_in=50.0,
        )

    """

    strategy_universe: TradingStrategyUniverse = console_context["strategy_universe"]
    pricing_model: PricingModel = console_context["pricing_model"]
    execution_model: EthereumExecution = console_context["execution_model"]
    routing_model: EthereumExecution = console_context["routing_model"]
    our_vault: LagoonVault = console_context["vault"]
    state: State = console_context["state"]
    web3: Web3 = console_context["web3"]
    store: JSONFileStore = console_context["store"]
    web3 = console_context["web3"]
    hot_wallet = execution_model.tx_builder.hot_wallet
    chain_id = web3.eth.chain_id

    in_token = fetch_erc20_details(
        web3,
        in_token,
    )

    out_token = fetch_erc20_details(
        web3,
        out_token,
    )

    amount = Decimal(amount_in)

    #
    # Before we start let's ask for a quote so we know CowSwap can fulfill
    # our swap before starting swapping, and we know there is a route
    # available.
    #
    quote = fetch_quote(
        from_=hot_wallet.address,  # Not deployed vault address yet, so use our hot wallet as a placeholder
        buy_token=out_token,
        sell_token=in_token,
        amount_in=amount,
        min_amount_out=Decimal(0),
        price_quality="verified",
    )
    print(f"Out CowSwap quote data is:\n{quote.pformat()}")

    print("Proceed with the swap [y/N]? ", end="")
    answer = input().strip().lower()
    if answer != "y":
        print("Aborting")
        return


    # 25% slippage max
    # We are doing swaps with very small amounts so we are getting
    # massive cost impact because fees are proportional to the swap size.


    # Don't do this: don't blindly trust the quote from CowSwap API,
    # verify price from onchain or offchain source.
    # Here we do this just for the example.
    # This is not the right way tp do this, because CoW Swap quoter already includes its slippage,
    # and we should only do this when using an external mid price as the price source.
    estimated_out = quote.get_buy_amount()
    slippaged_amount_out = estimated_out * Decimal(1 - max_slippage)

    print(f"Target price is {quote.get_price():.6f} {in_token.symbol}/{out_token.symbol}")
    print(f"We set the max slippage goal to {slippaged_amount_out:.6f} {out_token.symbol} for {slippaged_amount_out:.6f} {out_token.symbol} with max slippage of {max_slippage * 100:.1f}%")

    #
    # 5. Perform an automated Cowswap trade with the assets from the vault.
    # Swap all of out WETH to USDC.e via Cowswap integration.
    #

    # 5.a) The Gnosis Safe of the vault needs to approve the swap amount on the CowSwap settlement contract
    # deposit_request = deposit_manager.create_deposit_request(our_address, amount=usdc_amount)
    _broadcast_tx(
        hot_wallet,
        approve_cow_swap(
            vault=our_vault,
            token=in_token,
            amount=amount,
        ),
    )

    # 5.b) Create the presigned CowSwap order onchain via Lagoon vault TradingStrategyModuleV0
    # The order is createad onchain using SwapCowSwap._signCowSwapOrder() contract call.
    # We print the results to see what kind of order data we have created.

    _cowswap_broadcast_callback = lambda _web3, _hot_wallet, _bound_func: _broadcast_tx(_hot_wallet, _bound_func).hash

    order_data = presign_and_broadcast(
        asset_manager=hot_wallet,
        vault=our_vault,
        buy_token=out_token,
        sell_token=in_token,
        amount_in=amount,
        min_amount_out=slippaged_amount_out,
        broadcast_callback=_cowswap_broadcast_callback,
    )
    print(f"Our CoW Swap presigned order is:\n{pformat(order_data)}")

    print(f"View the order at CoW Swap explorer https://explorer.cow.fi/arb1/search/{order_data['uid']}")

    cowswap_result = execute_presigned_cowswap_order(
        chain_id=chain_id,
        order=order_data,
    )

    print(f"Cowswap order completed, order UID: {cowswap_result.order_uid.hex()}, status: {cowswap_result.get_status()}")

    status = cowswap_result.get_status()
    if status == "traded":
        # Make CowSwap sound effect
        print("Moooooo 🐮")
        print(f"Order final result:\n{pformat(cowswap_result.final_status_reply)}")
        print(f"All ok, check the vault at https://routescan.io/{our_vault.address}")
    else:
        print(f"Order failed - not sure why:\n{pformat(cowswap_result.final_status_reply)}")


