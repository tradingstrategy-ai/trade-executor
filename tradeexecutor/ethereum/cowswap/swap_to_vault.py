"""Open a vault position by swapping USDC to vault share tokens via CoW Swap.

Instead of using PositionManager which deposits through the ERC-4626 ``deposit()`` function,
this module buys vault share tokens directly on the open market through CoW Swap.
A tracked position and trade are created in the trade executor state.

Intended for vault share tokens that are traded on secondary markets, such as:

- Staked USDAi (sUSDAi)
- LLama Lend pools (future)

This is useful when:

- The vault has a deposit queue or settlement delay (e.g. Lagoon vaults)
- You want to acquire shares immediately without waiting for settlement
- The share token has sufficient DEX liquidity

**Requirements:**

- Lagoon vault with CoW Swap integration enabled via ``TradingStrategyModuleV0``
- The vault share token must have DEX liquidity that CoW Swap can route through

Console usage example:

.. code-block:: python

    from tradeexecutor.ethereum.cowswap.swap_to_vault import open_vault_position_cowswap

    open_vault_position_cowswap(
        locals(),
        vault_name="IPOR USDC Lending Optimizer",
        amount_usd=100.0,
    )
"""

import datetime
import logging
from decimal import Decimal
from pprint import pformat

from eth_defi.compat import native_datetime_utc_now
from eth_defi.confirmation import broadcast_and_wait_transactions_to_complete
from eth_defi.cow.quote import fetch_quote
from eth_defi.cow.status import CowSwapResult
from eth_defi.erc_4626.vault_protocol.lagoon.cowswap import (
    approve_cow_swap,
    execute_presigned_cowswap_order,
    presign_and_broadcast,
)
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.gas import apply_gas, estimate_gas_price
from eth_defi.hotwallet import HotWallet, SignedTransactionWithNonce
from eth_defi.token import fetch_erc20_details, TokenDetails
from web3 import Web3
from web3.contract.contract import ContractFunction

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.state.trade import TradeExecution, TradeFlag, TradeType
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


logger = logging.getLogger(__name__)


def _broadcast_tx(
    hot_wallet: HotWallet,
    bound_func: ContractFunction,
    default_gas_limit: int = 1_000_000,
) -> SignedTransactionWithNonce:
    """Broadcast a single transaction from the hot wallet."""
    web3 = bound_func.w3
    gas_price_suggestion = estimate_gas_price(web3)
    tx_params = apply_gas({}, gas_price_suggestion)

    if "gas" not in tx_params:
        tx_params["gas"] = default_gas_limit

    tx = hot_wallet.sign_bound_call_with_new_nonce(bound_func, tx_params=tx_params)
    logger.info("Broadcasting tx %s, calling %s()", tx.hash.hex(), bound_func.fn_name or "<unknown>")
    broadcast_and_wait_transactions_to_complete(web3, [tx])
    return tx


def _extract_executed_amounts(
    cowswap_result: CowSwapResult,
    sell_token: TokenDetails,
    buy_token: TokenDetails,
) -> tuple[Decimal, Decimal]:
    """Extract executed sell and buy amounts from a CoW Swap result.

    :return:
        Tuple of (executed_sell_amount, executed_buy_amount) in human-readable decimals.
    """
    status_data = cowswap_result.final_status_reply

    # The traded status reply contains executedAmounts in the value array
    if "value" in status_data and len(status_data["value"]) > 0:
        executed = status_data["value"][0].get("executedAmounts", {})
        raw_sell = int(executed.get("sell", 0))
        raw_buy = int(executed.get("buy", 0))
    else:
        # Fall back to order data amounts
        raw_sell = int(cowswap_result.order.get("sellAmount", 0))
        raw_buy = int(cowswap_result.order.get("buyAmount", 0))

    executed_sell = sell_token.convert_to_decimals(raw_sell)
    executed_buy = buy_token.convert_to_decimals(raw_buy)

    return executed_sell, executed_buy


def open_vault_position_cowswap(
    console_context: dict,
    vault_name: str,
    amount_usd: float,
    max_slippage: float = 0.01,
    notes: str | None = None,
) -> TradeExecution:
    """Open a tracked vault position by swapping USDC to vault share tokens via CoW Swap.

    Creates a proper position and trade in the trade executor state, then executes
    the swap through CoW Swap and marks the trade as successful.

    Call from the trade-executor console:

    .. code-block:: python

        from tradeexecutor.ethereum.cowswap.swap_to_vault import open_vault_position_cowswap

        open_vault_position_cowswap(
            locals(),
            vault_name="IPOR USDC Lending Optimizer",
            amount_usd=100.0,
            max_slippage=0.01,
        )

    :param console_context:
        Pass ``locals()`` from the console session.

    :param vault_name:
        Name of the vault to open a position in (resolved via ``strategy_universe.get_pair_by_vault_name()``).

    :param amount_usd:
        Dollar amount of USDC to swap to vault share tokens.

    :param max_slippage:
        Maximum slippage tolerance as a fraction (e.g. 0.01 = 1%).

    :param notes:
        Optional notes to attach to the trade.

    :return:
        The completed TradeExecution.

    :raises AssertionError:
        If the CoW Swap order fails or is cancelled.
    """

    strategy_universe: TradingStrategyUniverse = console_context["strategy_universe"]
    pricing_model: PricingModel = console_context["pricing_model"]
    our_vault: LagoonVault = console_context["vault"]
    state: State = console_context["state"]
    web3: Web3 = console_context["web3"]
    store: JSONFileStore = console_context["store"]
    hot_wallet: HotWallet = console_context["hot_wallet"]

    chain_id = web3.eth.chain_id
    ts = native_datetime_utc_now()

    # 1. Resolve the vault pair and token details
    pair: TradingPairIdentifier = strategy_universe.get_pair_by_vault_name(vault_name)
    reserve_asset = strategy_universe.get_reserve_asset()

    usdc = fetch_erc20_details(web3, pair.quote.address)
    share_token = fetch_erc20_details(web3, pair.base.address)

    amount = Decimal(str(amount_usd))

    if notes is None:
        notes = f"CoW Swap: {usdc.symbol} -> {share_token.symbol}"

    existing_position = state.portfolio.get_position_by_trading_pair(pair)
    action = "Increasing" if existing_position else "Opening"
    print(f"{action} vault position via CoW Swap")
    print(f"  Vault: {vault_name}")
    print(f"  Swap: {amount} {usdc.symbol} -> {share_token.symbol}")
    print(f"  Share token: {share_token.address}")
    print(f"  Max slippage: {max_slippage * 100:.1f}%")

    # 2. Fetch quote from CoW Swap to check route and get expected output
    quote = fetch_quote(
        from_=hot_wallet.address,
        buy_token=share_token,
        sell_token=usdc,
        amount_in=amount,
        min_amount_out=Decimal(0),
        price_quality="verified",
    )

    estimated_shares = quote.get_buy_amount()
    min_amount_out = estimated_shares * Decimal(1 - max_slippage)
    estimated_price = float(amount / estimated_shares)

    print(f"  Quote: ~{estimated_shares:.6f} {share_token.symbol}")
    print(f"  Estimated price: {estimated_price:.6f} {usdc.symbol}/{share_token.symbol}")
    print(f"  Min output (with slippage): {min_amount_out:.6f} {share_token.symbol}")

    # 3. Create tracked trade and position in state
    flags = {TradeFlag.increase} if existing_position else {TradeFlag.open}

    position, trade, created = state.create_trade(
        strategy_cycle_at=ts,
        pair=pair,
        quantity=None,
        reserve=amount,
        assumed_price=estimated_price,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes=notes,
        slippage_tolerance=max_slippage,
        flags=flags,
    )

    if created:
        print(f"  Created new position #{position.position_id}")
    else:
        print(f"  Increasing position #{position.position_id}")

    # 4. Start execution - moves capital from reserves to the trade
    state.start_execution(
        ts=ts,
        trade=trade,
        underflow_check=True,
    )

    # Mark as broadcasted so mark_trade_success works
    state.mark_broadcasted(ts, trade)

    # 5. Execute the CoW Swap
    hot_wallet.sync_nonce(web3)

    # 5a. Approve CoW Swap vault relayer to spend USDC from the vault Safe
    print("  Approving CoW Swap vault relayer...")
    _broadcast_tx(
        hot_wallet,
        approve_cow_swap(
            vault=our_vault,
            token=usdc,
            amount=amount,
        ),
    )

    # 5b. Create and broadcast the presigned order onchain
    print("  Creating presigned CoW Swap order...")
    _cowswap_broadcast_callback = lambda _web3, _hw, _func: _broadcast_tx(_hw, _func).hash

    order_data = presign_and_broadcast(
        asset_manager=hot_wallet,
        vault=our_vault,
        buy_token=share_token,
        sell_token=usdc,
        amount_in=amount,
        min_amount_out=min_amount_out,
        broadcast_callback=_cowswap_broadcast_callback,
    )
    print(f"  Order UID: {order_data['uid']}")

    # 5c. Post order to CoW Swap API and wait for settlement
    print("  Waiting for CoW Swap settlement...")
    cowswap_result = execute_presigned_cowswap_order(
        chain_id=chain_id,
        order=order_data,
    )

    status = cowswap_result.get_status()
    assert status == "traded", f"CoW Swap order failed with status: {status}\n{pformat(cowswap_result.final_status_reply)}"

    # 6. Extract executed amounts and mark trade as successful
    executed_reserve, executed_shares = _extract_executed_amounts(
        cowswap_result,
        sell_token=usdc,
        buy_token=share_token,
    )

    if executed_reserve == Decimal(0) or executed_shares == Decimal(0):
        # Fallback: use planned amounts if execution data not in status reply
        executed_reserve = amount
        executed_shares = estimated_shares

    executed_price = float(executed_reserve / executed_shares)

    state.mark_trade_success(
        executed_at=native_datetime_utc_now(),
        trade=trade,
        executed_price=executed_price,
        executed_amount=executed_shares,
        executed_reserve=executed_reserve,
        lp_fees=0,
        native_token_price=0,
    )

    # 7. Save state
    store.sync(state)

    print(f"  Position #{position.position_id} opened successfully")
    print(f"  Executed: {executed_shares:.6f} {share_token.symbol} for {executed_reserve:.6f} {usdc.symbol}")
    print(f"  Price: {executed_price:.6f} {usdc.symbol}/{share_token.symbol}")

    return trade
