import datetime
import time
from typing import List, Dict

from eth_account.signers.local import LocalAccount
from eth_typing import HexAddress
from hexbytes import HexBytes
from web3 import Web3

from smart_contracts_for_testing.gas import GasPriceSuggestion, apply_gas, estimate_gas_fees
from smart_contracts_for_testing.token import fetch_erc20_details
from smart_contracts_for_testing.txmonitor import wait_transactions_to_complete
from smart_contracts_for_testing.uniswap_v2 import UniswapV2Deployment, FOREVER_DEADLINE
from tradeexecutor.state.state import TradeExecution, State, BlockchainTransactionInfo


def translate_to_naive_swap(web3: Web3, deployment: UniswapV2Deployment, hot_wallet: LocalAccount, t: TradeExecution, nonce: int, gas_fees: GasPriceSuggestion):
    """Creates an AMM swap tranasction out of buy/sell.

    If buy tries to do the best execution for given `planned_reserve`.

    If sell tries to do the best execution for given `planned_quantity`.

    Route only between two pools - stablecoin reserve and target buy/sell.

    Any gas price is set by `web3` instance gas price strategy.

    :param t:
    :return: Unsigned transaction
    """

    base_token_details = fetch_erc20_details(web3, t.pair.base.address)
    quote_token_details = fetch_erc20_details(web3, t.pair.quote.address)

    assert base_token_details.decimals is not None, f"Bad token at {t.pair.base.address}"
    assert quote_token_details.decimals is not None, f"Bad token at {t.pair.quote.address}"

    if t.is_buy():
        amount0_in = int(t.planned_reserve * 10**quote_token_details.decimals)
        path = [quote_token_details.address, base_token_details.address]
    else:
        # Reverse swap
        amount0_in = int(t.planned_quantity * 10**base_token_details.decimals)
        path = [base_token_details.address, quote_token_details.address]

    args = [
        amount0_in,
        0,
        path,
        hot_wallet.address,
        FOREVER_DEADLINE,
    ]

    # https://docs.uniswap.org/protocol/V2/reference/smart-contracts/router-02#swapexacttokensfortokens
    # https://web3py.readthedocs.io/en/stable/web3.eth.account.html#sign-a-contract-transaction
    tx = deployment.router.functions.swapExactTokensForTokens(
        *args,
    ).buildTransaction({
        'chainId': 1,
        'gas': 350_000,  # Estimate max 350k gas per swap
        'nonce': nonce,
        'from': hot_wallet.address,
    })

    apply_gas(tx, gas_fees)

    tx_info = t.tx_info = BlockchainTransactionInfo()

    selector = deployment.router.functions.swapExactTokensForTokens
    tx_info.function_selector = selector.fn_name
    tx_info.args = args
    tx_info.details = tx
    tx_info.nonce = nonce


def prepare_swaps(
        web3: Web3,
        hot_wallet: LocalAccount,
        uniswap: UniswapV2Deployment,
        ts: datetime.datetime,
        state: State,
        instructions: List[TradeExecution]):
    """Prepare multiple swaps to be breoadcasted parallel from the hot wallet."""

    # Get our starting nonce
    start_nonce = web3.eth.get_transaction_count(hot_wallet.address)
    gas_fees = estimate_gas_fees(web3)

    for idx, t in enumerate(instructions):
        nonce = start_nonce + idx

        state.portfolio.check_for_nonce_reuse(nonce)

        translate_to_naive_swap(
            web3,
            uniswap,
            hot_wallet,
            t,
            nonce,
            gas_fees,
        )
        signed = hot_wallet.sign_transaction(t.tx_info.details)
        t.tx_info.signed_bytes = signed.rawTransaction
        t.tx_info.tx_hash = signed.hash
        t.started_at = ts


def broadcast(
        web3: Web3,
        ts: datetime.datetime,
        instructions: List[TradeExecution]) -> Dict[HexBytes, TradeExecution]:
    """Broadcast multiple transations.

    :return: Map of transaction hashes to watch
    """
    res = {}
    for t in instructions:
        assert isinstance(t.tx_info.signed_bytes, HexBytes), f"Got signed transaction: {t.tx_info.signed_bytes}"
        web3.eth.send_raw_transaction(t.tx_info.signed_bytes)
        t.broadcasted_at = ts
        res[t.tx_info.tx_hash] = t
    return res


def wait_trades_to_complete(
        web3: Web3,
        trades: List[TradeExecution],
        max_timeout=datetime.timedelta(minutes=5),
        poll_delay=datetime.timedelta(seconds=1)) -> Dict[HexBytes, dict]:
    """Watch multiple transactions executed at parallel.

    :return: Map of transaction hashes -> receipt
    """
    tx_hashes = [t.tx_info.tx_hash for t in trades]
    receipts = wait_transactions_to_complete(web3, tx_hashes, max_timeout, poll_delay)
    return receipts


def resolve_trades(state: State, ts: datetime.datetime, trades: Dict[HexBytes, TradeExecution], receipts: Dict[HexBytes, dict]):
    """Resolve trade outcome."""

    for tx_hash, receipt in receipts.items():
        trade = trades[tx_hash]
        if receipt.status == 1:
            # Mark as success
            state.mark_trade_success(
                ts,
                trade,
                executed_price=0,
                executed_amount=0,
                executed_reserve=0,
                lp_fees=0,
                gas_price=0,
                gas_used=0,
                native_token_price=0,
            )
        else:
            # Mark as reverted
            pass

