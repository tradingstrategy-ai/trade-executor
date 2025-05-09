"""Route trades for ERC-4626 and similar vaults."""

import logging
from _decimal import Decimal
from typing import Dict, cast

from eth_typing import HexAddress
from hexbytes import HexBytes

from eth_defi.erc_4626.analysis import analyse_4626_flow_transaction
from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.flow import approve_and_deposit_4626, approve_and_redeem_4626
from eth_defi.erc_4626.vault import ERC4626Vault
from eth_defi.token import fetch_erc20_details
from eth_defi.trade import TradeSuccess


from tradeexecutor.ethereum.swap import get_swap_transactions, report_failure
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import JSONHexAddress
from tradeexecutor.strategy.routing import RoutingState, RoutingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.utils.blockchain import get_block_timestamp
from web3 import Web3

from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse


logger = logging.getLogger(__name__)


class VaultRoutingState(RoutingState):
    """Capture trade executor state what we need for one strategy cycle of ERC-4626 deposits and redeems.

    - Not much to do here - Enso swaps are stateless (no approves needed)
    """

    def __init__(
        self,
        tx_builder: TransactionBuilder,
        strategy_universe: TradingStrategyUniverse,
    ):
        self.tx_builder = tx_builder
        self.strategy_universe = strategy_universe

    def get_reserve_asset(self) -> AssetIdentifier:
        return self.strategy_universe.get_reserve_asset()


class VaultRouting(RoutingModel):
    """ERC-4626 routing.

    - Do trades for ERC-4626 and other vaults
    """

    def __init__(self, reserve_token_address: JSONHexAddress):
        super().__init__(
            allowed_intermediary_pairs={},
            reserve_token_address=reserve_token_address,
        )

    def create_routing_state(
        self,
        universe: StrategyExecutionUniverse,
        execution_details: dict
    ) -> VaultRoutingState:
        return VaultRoutingState(
            tx_builder=execution_details["tx_builder"],
            strategy_universe=cast(TradingStrategyUniverse, universe),
        )

    def perform_preflight_checks_and_logging(self,
        pair_universe: PandasPairUniverse):
        """"Checks the integrity of the routing.

        - Called from check-wallet to see our routing and balances are good
        """
        logger.info("Routing details")
        self.reserve_asset_logging(pair_universe)

    def deposit_or_redeem(
        self,
        state: VaultRoutingState,
        trade: TradeExecution,
    ) -> list[BlockchainTransaction]:
        """Prepare vault flow transactions."""

        assert trade.is_vault(), "Vault only supports vault trades"
        assert trade.slippage_tolerance, "TradeExecution.slippage_tolerance must be set"
        assert trade.pair.quote.address in self.allowed_intermediary_pairs or trade.pair.quote.address == self.reserve_token_address, f"Unsupported quote token: {trade.pair}"

        reserve_asset = state.strategy_universe.get_reserve_asset()

        tx_builder = state.tx_builder
        web3 = tx_builder.web3

        target_vault = get_vault_for_pair(web3, trade.pair)

        if trade.is_buy():
            token_in = reserve_asset
            token_out = trade.pair.base
            swap_amount = trade.get_planned_reserve()
        else:
            token_in = trade.pair.base
            token_out = reserve_asset
            swap_amount = -trade.planned_quantity

        logger.info(
            "Preparing vault flow %s -> %s, amount %s (%s), slippage tolerance %f",
            token_in.token_symbol,
            token_out.token_symbol,
            swap_amount,
            token_in.convert_to_decimal(swap_amount),
            trade.slippage_tolerance,
        )

        asset_deltas = trade.calculate_asset_deltas()
        address = HexAddress(tx_builder.get_token_delivery_address())

        if trade.is_buy():
            approve_call, swap_call = approve_and_deposit_4626(
                vault=target_vault,
                from_=address,
                amount=swap_amount
            )
        else:
            approve_call, swap_call = approve_and_redeem_4626(
                vault=target_vault,
                from_=address,
                amount=swap_amount
            )

        approve_gas_limit = 500_000
        swap_gas_limit = 2_500_000

        tx_1 = tx_builder.sign_transaction(
            contract=target_vault.vault_contract,
            args_bound_func=approve_call,
            gas_limit=approve_gas_limit,
            asset_deltas=[],
            notes=trade.notes,
        )

        tx_2 = tx_builder.sign_transaction(
            contract=target_vault.vault_contract,
            args_bound_func=swap_call,
            gas_limit=swap_gas_limit,
            asset_deltas=[],
            notes=trade.notes,
        )
        return [tx_1, tx_2]

    def setup_trades(
        self,
        state: State,
        routing_state: VaultRoutingState,
        trades: list[TradeExecution],
        check_balances=False,
        rebroadcast=False,
    ):
        """
        See test_velvet_e2e for tests.

        Error codes:

        - Revert reason: execution reverted: custom error 0xe2f23246

        - 2Po: Enso slippage error, or out of funds
        """

        logger.info(
            "Preparing %d trades for ERC-4626 execution",
            len(trades),
        )

        for trade in trades:
            assert trade.is_vault(), f"Not a vault trade: {trade}"
            trade.blockchain_transactions = self.deposit_or_redeem(routing_state, trade)

    def settle_trade(
        self,
        web3: Web3,
        state: State,
        trade: TradeExecution,
        receipts: Dict[str, dict],
        stop_on_execution_failure=False,
    ):

        vault = get_vault_for_pair(web3, trade.pair)
        logger.info(f"Settling vault trade: #{trade.trade_id} for {vault}")

        base_token_details = fetch_erc20_details(web3, trade.pair.base.checksum_address)
        # quote_token_details = fetch_erc20_details(web3, trade.pair.quote.checksum_address)
        reserve = trade.reserve_currency

        swap_tx = get_swap_transactions(trade)

        try:
            receipt = receipts[HexBytes(swap_tx.tx_hash)]
        except KeyError as e:
            raise KeyError(f"Could not find hash: {swap_tx.tx_hash} in {receipts}") from e

        direction = "deposit" if trade.is_buy() else "redeem"
        result = analyse_4626_flow_transaction(
            vault=vault,
            tx_hash=swap_tx.tx_hash,
            tx_receipt=receipt,
            direction=direction,
        )

        ts = get_block_timestamp(web3, receipt["blockNumber"])

        if isinstance(result, TradeSuccess):

            path = result.path

            if trade.is_buy():
                assert path[0] == reserve.address, f"Was expecting the route path to start with reserve token {reserve}, got path {result.path}"

                # price = result.get_human_price(quote_token_details.address == result.token0.address)
                executed_reserve = result.amount_in / Decimal(10 ** reserve.decimals)
                executed_amount = result.amount_out / Decimal(10 ** base_token_details.decimals)

                price = executed_reserve / executed_amount

            else:
                # Ordered other way around
                assert path[0] == base_token_details.address.lower(), f"Path is {path}, base token is {base_token_details}"
                assert path[-1] == reserve.address
                # price = result.get_human_price(quote_token_details.address == result.token0.address)
                executed_amount = -result.amount_in / Decimal(10 ** base_token_details.decimals)
                executed_reserve = result.amount_out / Decimal(10 ** reserve.decimals)
                price = -executed_reserve / executed_amount

            assert (executed_reserve > 0) and (executed_amount != 0) and (price > 0), f"Executed amount {executed_amount}, executed_reserve: {executed_reserve}, price: {price}"

            logger.info("Executed amount: %s, executed reserve: %s, price: %s", executed_amount, executed_reserve, price)

            # Mark as success
            state.mark_trade_success(
                ts,
                trade,
                executed_price=float(price),
                executed_amount=executed_amount,
                executed_reserve=executed_reserve,
                lp_fees=0,
                native_token_price=0,  # won't fix
                cost_of_gas=float(result.get_cost_of_gas()),
            )

            slippage = trade.get_slippage()
            logger.info(f"Executed: {executed_amount} {trade.pair.base.token_symbol}, {executed_reserve} {trade.pair.quote.token_symbol}, price: {trade.executed_price}, expected reserve: {trade.planned_reserve} {trade.pair.quote.token_symbol}, slippage {slippage:.2%}")

        else:
            # Trade failed
            report_failure(ts, state, trade, stop_on_execution_failure)


def get_vault_for_pair(
    web3: Web3,
    target_pair: TradingPairIdentifier,
) -> ERC4626Vault:
    """Get a cached Vault instance based on a trading pair.

    - Instance has a web3 connection object
    """

    assert target_pair.is_vault()

    vault_address = target_pair.pool_address
    features = target_pair.get_vault_features()

    assert features, f"Vault features missing: {target_pair}"

    cache_key = (vault_address, id(web3))
    cached = _vault_cache.get(cache_key)
    if cached:
        return cached

    cached = create_vault_instance(web3, vault_address, features)
    _vault_cache[cache_key] = cached
    return cached


_vault_cache = {}
