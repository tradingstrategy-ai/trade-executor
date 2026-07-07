"""Route trades for ERC-4626 and similar vaults."""

import logging
from decimal import Decimal
import datetime
from typing import Dict, cast

from eth_typing import HexAddress
from hexbytes import HexBytes

from eth_defi.erc_4626.analysis import analyse_4626_flow_transaction
from eth_defi.erc_4626.classification import create_vault_instance, create_vault_instance_autodetect
from eth_defi.erc_4626.flow import approve_and_deposit_4626, approve_and_redeem_4626
from eth_defi.erc_4626.profit_and_loss import estimate_4626_recent_profitability
from eth_defi.erc_4626.vault import ERC4626Vault
from eth_defi.token import fetch_erc20_details, TokenDiskCache
from eth_defi.trade import TradeSuccess
from eth_defi.vault.deposit_redeem import VaultDepositManager

from tradeexecutor.ethereum.swap import get_swap_transactions, report_failure
from tradeexecutor.ethereum.token_cache import get_default_token_cache
from tradeexecutor.ethereum.vault.settlement_estimate import refresh_vault_settlement_estimate
from tradeexecutor.ethereum.vault.vault_utils import is_explicit_generic_erc4626_pair
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.portfolio import Portfolio
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
        token_cache: TokenDiskCache | None = None,
    ):
        self.tx_builder = tx_builder
        self.strategy_universe = strategy_universe
        self.token_cache = token_cache

    def get_reserve_asset(self) -> AssetIdentifier:
        return self.strategy_universe.get_reserve_asset()


class VaultRouting(RoutingModel):
    """ERC-4626 routing.

    - Do trades for ERC-4626 and other vaults
    """

    def __init__(
        self,
        reserve_token_address: JSONHexAddress,
        profitability_estimation_lookback_window=datetime.timedelta(days=7),
        epsilon=Decimal(1e-6),
        redeem_epsilon=0.025,
    ):
        super().__init__(
            allowed_intermediary_pairs={},
            reserve_token_address=reserve_token_address,
        )
        self.profitability_estimation_lookback_window = profitability_estimation_lookback_window
        self.epsilon = epsilon

        # 3M gas was not enough to withdraw from IPOR, but Base has a per-tx gas cap 16,777,216
        self.vault_interaction_gas_limit = 10_000_000

        # 2.5% is the maximum relative difference for redeeming vault shares,
        # when checking onchain balance vs our internal accounting
        self.redeem_epsilon = redeem_epsilon
        self.token_cache: TokenDiskCache | None = None

    def create_routing_state(
        self,
        universe: StrategyExecutionUniverse,
        execution_details: dict
    ) -> VaultRoutingState:
        self.token_cache = execution_details.get("token_cache")
        return VaultRoutingState(
            tx_builder=execution_details["tx_builder"],
            strategy_universe=cast(TradingStrategyUniverse, universe),
            token_cache=self.token_cache,
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
        state: State,
        routing_state: VaultRoutingState,
        trade: TradeExecution,
    ) -> list[BlockchainTransaction]:
        """Prepare vault flow transactions."""

        assert isinstance(state, State)
        assert isinstance(routing_state, VaultRoutingState)

        assert trade.is_vault(), "Vault only supports vault trades"
        assert trade.slippage_tolerance, "TradeExecution.slippage_tolerance must be set"

        reserve_asset = routing_state.strategy_universe.get_reserve_asset()

        # Cross-chain vault trades use the satellite chain's reserve token
        # (e.g. Base USDC) which differs from the home chain reserve (Arb USDC).
        # The on-chain deposit uses the vault's own denomination_token regardless.
        if trade.pair.quote.chain_id == reserve_asset.chain_id:
            assert trade.pair.quote.address in self.allowed_intermediary_pairs or trade.pair.quote.address == self.reserve_token_address, f"Unsupported quote token: {trade.pair}: {trade.pair.quote.address}, our reserve is {self.reserve_token_address}"

        tx_builder = routing_state.tx_builder
        web3 = tx_builder.web3
        address = HexAddress(tx_builder.get_token_delivery_address())

        target_vault = get_vault_for_pair(
            web3,
            trade.pair,
            token_cache=routing_state.token_cache,
        )

        if trade.is_buy():
            token_in = reserve_asset
            token_out = trade.pair.base
            swap_amount = trade.get_planned_reserve()

            try:
                profitability_estimation = estimate_4626_recent_profitability(
                    vault=target_vault,
                    lookback_window=self.profitability_estimation_lookback_window,
                )
                profitability_estimation_error = None
            except Exception as e:
                # Ok to fail, data used only for diagnostics and UI
                profitability_estimation = None
                profitability_estimation_error = str(e)
                logger.error(
                    "Vault trade %s profitatability estimation failed: %s",
                )
        else:
            token_in = trade.pair.base
            token_out = reserve_asset
            # Swap amount is negative
            swap_amount = -trade.planned_quantity

            share_token = target_vault.share_token
            onchain_balance = share_token.fetch_balance_of(address)

            portfolio: Portfolio = state.portfolio
            position = portfolio.get_position_by_id(trade.position_id)
            share_token = trade.pair.base

            logger.info(
                "Vault redeem. Position quantity %s, trade quantity %s, onchain balance %s, position planned quantity %s",
                position.get_quantity(),
                trade.planned_quantity,
                onchain_balance,
                position.get_quantity(planned=True),
            )
            rel_diff = abs((onchain_balance - swap_amount) / swap_amount)
            if rel_diff != 0 and onchain_balance + swap_amount < 0:
                if rel_diff > self.redeem_epsilon:
                    # Accounting broken

                    logger.error(
                        "Vault trade %s, position %s, share token %s, has a large relative difference in onchain balance: %f, planned quantity: %s, onchain balance: %s, epsilon is %f",
                        trade.trade_id,
                        position,
                        share_token,
                        rel_diff,
                        trade.planned_quantity,
                        onchain_balance,
                        self.redeem_epsilon,
                    )
                    raise AssertionError("Vault share token has a large relative difference in onchain balance when trying to redeem the share token")
                else:
                    # Epsilon rounding
                    logger.warning(
                        "Vault trade %s, position %s, share token %s, has a small relative difference in onchain balance: %f, planned quantity: %s, onchain balance: %s, automatically rounding, epsilon is %f",
                        trade.trade_id,
                        position,
                        share_token,
                        rel_diff,
                        trade.planned_quantity,
                        onchain_balance,
                        self.redeem_epsilon,
                    )
                    swap_amount = onchain_balance
            else:
                # Exact match
                logger.info("Onchain balance and accounting has exact match for shares to redeem: %s", swap_amount)

        logger.info(
            "Preparing vault flow %s -> %s, amount %s (%s), slippage tolerance %f",
            token_in.token_symbol,
            token_out.token_symbol,
            swap_amount,
            token_in.convert_to_decimal(swap_amount),
            trade.slippage_tolerance,
        )

        asset_deltas = trade.calculate_asset_deltas()

        deposit_manager = target_vault.get_deposit_manager()

        # Async vault flow (Ostium V1.5, ERC-7540 Lagoon)
        if trade.is_buy() and not deposit_manager.has_synchronous_deposit():
            return self._build_async_deposit_txs(
                tx_builder, target_vault, deposit_manager, trade, address, swap_amount,
            )
        elif trade.is_sell() and not deposit_manager.has_synchronous_redemption():
            return self._build_async_redeem_txs(
                tx_builder, target_vault, deposit_manager, trade, address, swap_amount,
            )

        # Synchronous vault flow (standard ERC-4626)
        if trade.is_buy():
            approve_call, swap_call = approve_and_deposit_4626(
                vault=target_vault,
                from_=address,
                amount=swap_amount,
                check_enough_token=False,
            )
        else:
            approve_call, swap_call = approve_and_redeem_4626(
                vault=target_vault,
                from_=address,
                amount=swap_amount
            )

        approve_gas_limit = 500_000
        swap_gas_limit = self.vault_interaction_gas_limit

        tx_1 = tx_builder.sign_transaction(
            contract=target_vault.denomination_token.contract,
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

    def _build_async_deposit_txs(
        self,
        tx_builder: TransactionBuilder,
        target_vault: ERC4626Vault,
        deposit_manager: VaultDepositManager,
        trade: TradeExecution,
        address: HexAddress,
        swap_amount: Decimal,
    ) -> list[BlockchainTransaction]:
        """Build approve + requestDeposit transactions for async vault deposit."""

        deposit_request = deposit_manager.create_deposit_request(
            owner=address,
            amount=swap_amount,
        )
        approve_call = target_vault.denomination_token.approve(
            target_vault.vault_address,
            swap_amount,
        )

        # Mark trade for async handling — store both raw and decimal amounts
        # so we can reconstruct the request during settle_trade() parsing.
        # Raw amounts are stored as strings: 18-decimal values exceed the
        # JavaScript safe-integer limit enforced by the state file validator.
        trade.other_data["vault_async_flow"] = True
        trade.other_data["vault_raw_amount"] = str(deposit_request.raw_amount)
        trade.other_data["vault_deposit_amount"] = str(swap_amount)
        trade.other_data["vault_owner_address"] = address

        # Estimate when this async deposit will settle, so the trade-ui can show
        # an ETA while the request is in vault_settlement_pending. Ostium V1.5
        # returns a real timestamp; operator-driven ERC-7540 vaults (Lagoon)
        # return None (no deterministic on-chain schedule).
        try:
            settles_at = deposit_manager.get_deposit_delay_over(address)
        except Exception as e:
            logger.warning("Could not estimate vault deposit settlement time for %s: %s", target_vault.vault_address, e)
            settles_at = None
        trade.other_data["vault_settlement_estimated_at"] = settles_at.isoformat() if settles_at else None

        # Sign approve tx first
        txs = [tx_builder.sign_transaction(
            contract=target_vault.denomination_token.contract,
            args_bound_func=approve_call,
            gas_limit=500_000,
            asset_deltas=[],
            notes=trade.notes,
        )]

        # Sign all request funcs (most adapters have 1, but support multiple)
        for func in deposit_request.funcs:
            txs.append(tx_builder.sign_transaction(
                contract=target_vault.vault_contract,
                args_bound_func=func,
                gas_limit=self.vault_interaction_gas_limit,
                asset_deltas=[],
                notes=trade.notes,
            ))

        trade.other_data["vault_request_tx_count"] = len(txs)
        return txs

    def _build_async_redeem_txs(
        self,
        tx_builder: TransactionBuilder,
        target_vault: ERC4626Vault,
        deposit_manager: VaultDepositManager,
        trade: TradeExecution,
        address: HexAddress,
        swap_amount: Decimal,
    ) -> list[BlockchainTransaction]:
        """Build requestWithdraw transaction for async vault redemption."""

        redemption_request = deposit_manager.create_redemption_request(
            owner=address,
            shares=swap_amount,
        )

        # Mark trade for async handling — store both raw and decimal amounts.
        # We store vault_redeem_shares (Decimal) for adapters like Lagoon that
        # assert `not raw_shares` and require the decimal form for reconstruction.
        # Raw amounts are stored as strings: 18-decimal share counts exceed the
        # JavaScript safe-integer limit enforced by the state file validator.
        trade.other_data["vault_async_flow"] = True
        trade.other_data["vault_raw_amount"] = str(redemption_request.raw_shares)
        trade.other_data["vault_redeem_shares"] = str(swap_amount)
        trade.other_data["vault_owner_address"] = address

        # Sign all request funcs (most adapters have 1, but support multiple)
        txs = []
        for func in redemption_request.funcs:
            txs.append(tx_builder.sign_transaction(
                contract=target_vault.vault_contract,
                args_bound_func=func,
                gas_limit=self.vault_interaction_gas_limit,
                asset_deltas=[],
                notes=trade.notes,
            ))

        trade.other_data["vault_request_tx_count"] = len(txs)
        return txs

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
            trade.blockchain_transactions = self.deposit_or_redeem(state, routing_state, trade)

    def settle_trade(
        self,
        web3: Web3,
        state: State,
        trade: TradeExecution,
        receipts: Dict[str, dict],
        stop_on_execution_failure=False,
    ):

        vault = get_vault_for_pair(
            web3,
            trade.pair,
            token_cache=self.token_cache,
        )
        logger.info(f"Settling vault trade: #{trade.trade_id} for {vault}")

        swap_tx = get_swap_transactions(trade)

        try:
            receipt = receipts[HexBytes(swap_tx.tx_hash)]
        except KeyError as e:
            raise KeyError(f"Could not find hash: {swap_tx.tx_hash} in {receipts}") from e

        ts = get_block_timestamp(web3, receipt["blockNumber"])

        # Async vault flow — parse request event and mark as pending settlement
        if trade.other_data.get("vault_async_flow"):
            if receipt["status"] == 0:
                report_failure(ts, state, trade, stop_on_execution_failure)
                return

            deposit_manager = vault.get_deposit_manager()
            direction = trade.other_data.get("vault_direction", "deposit" if trade.is_buy() else "redeem")
            owner_address = HexAddress(trade.other_data["vault_owner_address"])
            tx_hashes = [HexBytes(tx.tx_hash) for tx in trade.blockchain_transactions if tx.tx_hash]

            if direction == "deposit":
                # Reconstruct deposit request using raw_amount (int) —
                # all adapters support this path for deposits. int() accepts
                # both the current string form and the legacy int form.
                deposit_request = deposit_manager.create_deposit_request(
                    owner=owner_address,
                    raw_amount=int(trade.other_data["vault_raw_amount"]),
                )
                ticket = deposit_request.parse_deposit_transaction(tx_hashes)
                ticket_data = deposit_manager.serialize_deposit_ticket(ticket)
                refresh_vault_settlement_estimate(
                    trade,
                    deposit_manager,
                    ticket,
                    direction,
                )
            else:
                # Reconstruct redemption request using shares (Decimal) —
                # Lagoon asserts `not raw_shares` so we must pass the decimal form.
                # Fall back to raw_shares for adapters that only support raw form.
                # check_enough_token=False: the real requestRedeem() already moved
                # the shares to the vault escrow, so the owner balance now reads zero;
                # we only rebuild the request to parse the broadcast transaction.
                redeem_shares_str = trade.other_data.get("vault_redeem_shares")
                if redeem_shares_str:
                    redemption_request = deposit_manager.create_redemption_request(
                        owner=owner_address,
                        shares=Decimal(redeem_shares_str),
                        check_enough_token=False,
                    )
                else:
                    # Legacy path: older trades stored only vault_raw_amount
                    redemption_request = deposit_manager.create_redemption_request(
                        owner=owner_address,
                        raw_shares=int(trade.other_data["vault_raw_amount"]),
                        check_enough_token=False,
                    )
                ticket = redemption_request.parse_redeem_transaction(tx_hashes)
                ticket_data = deposit_manager.serialize_redemption_ticket(ticket)
                refresh_vault_settlement_estimate(
                    trade,
                    deposit_manager,
                    ticket,
                    direction,
                )

            state.mark_vault_settlement_pending(ts, trade, ticket_data)
            logger.info(
                "Vault trade #%d marked as settlement pending (direction=%s, ticket=%s)",
                trade.trade_id, direction, ticket_data,
            )
            return

        # Synchronous vault flow — analyse the deposit/redeem result
        base_token_details = fetch_erc20_details(
            web3,
            trade.pair.base.checksum_address,
            cache=self.token_cache,
            chain_id=trade.pair.base.chain_id,
        )
        reserve = trade.reserve_currency
        direction = "deposit" if trade.is_buy() else "redeem"

        try:
            result = analyse_4626_flow_transaction(
                vault=vault,
                tx_hash=swap_tx.tx_hash,
                tx_receipt=receipt,
                direction=direction,
                hot_wallet=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to analyse vault tx: {swap_tx.wrapped_function_selector}: {swap_tx.tx_hash} direction: {direction}, receipt: {receipt}, vault: {vault}") from e

        if isinstance(result, TradeSuccess):

            path = result.path

            # For cross-chain vault trades the on-chain path uses the
            # satellite chain's quote token (e.g. Base USDC) whereas
            # reserve_currency is the home chain token (Arb USDC).
            expected_quote_addr = trade.pair.quote.address.lower()

            if trade.is_buy():
                assert path[0] == expected_quote_addr, f"Was expecting the route path to start with quote token {trade.pair.quote}, got path {result.path}"

                executed_reserve = result.amount_in / Decimal(10 ** reserve.decimals)
                executed_amount = result.amount_out / Decimal(10 ** base_token_details.decimals)

                price = executed_reserve / executed_amount

            else:
                assert path[0] == base_token_details.address.lower(), f"Path is {path}, base token is {base_token_details}"
                assert path[-1] == expected_quote_addr, f"Path is {path}, expected quote token {trade.pair.quote}"
                executed_amount = -result.amount_in / Decimal(10 ** base_token_details.decimals)
                executed_reserve = result.amount_out / Decimal(10 ** reserve.decimals)
                price = -executed_reserve / executed_amount

            assert (executed_reserve > 0) and (executed_amount != 0) and (price > 0), f"Executed amount {executed_amount}, executed_reserve: {executed_reserve}, price: {price}"

            logger.info("Executed amount: %s, executed reserve: %s, price: %s", executed_amount, executed_reserve, price)

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
    token_cache: "TokenDiskCache | None" = None,
) -> ERC4626Vault:
    """Get a cached Vault instance based on a trading pair.

    - Instance has a web3 connection object

    :param token_cache:
        Token cache for caching ERC-20 token metadata.
        If not provided, uses default cache.
    """

    assert target_pair.is_vault()

    vault_address = target_pair.pool_address
    features = target_pair.get_vault_features()

    cache_key = (vault_address, id(web3))
    cached = _vault_cache.get(cache_key)

    if token_cache is None:
        token_cache = get_default_token_cache()

    if cached:
        return cached

    if features or is_explicit_generic_erc4626_pair(target_pair):
        cached = create_vault_instance(
            web3,
            vault_address,
            features or set(),
            token_cache=token_cache,
        )
    else:
        # Autodetect features, much slower
        cached = create_vault_instance_autodetect(
            web3,
            vault_address,
            token_cache=token_cache,
        )

    _vault_cache[cache_key] = cached
    return cached


#: In-process cache of constructed vault objects
_vault_cache = {}
