"""Route trades for ERC-4626 and similar vaults."""

import logging
from decimal import Decimal
from typing import cast

from eth_typing import HexAddress
from hexbytes import HexBytes

from eth_defi.token import USDC_NATIVE_TOKEN

from eth_defi.erc_4626.analysis import analyse_4626_flow_transaction
from eth_defi.erc_4626.classification import create_vault_instance, create_vault_instance_autodetect
from eth_defi.erc_4626.deposit_redeem import ERC4626DepositManager
from eth_defi.erc_4626.vault import ERC4626Vault
from eth_defi.token import fetch_erc20_details, TokenDiskCache
from eth_defi.trade import TradeSuccess
from eth_defi.vault.deposit_redeem import (
    DepositRedeemEventAnalysis,
    DepositRedeemEventFailure,
)

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


class VaultReceiptAnalysisError(RuntimeError):
    """A mined vault transaction could not be decoded into executed amounts."""


class IncompatibleDepositAsset(Exception):
    """The selected deposit asset is not accepted by a multi-asset vault.

    Multi-asset vaults (e.g. Upshift) whitelist a specific set of input assets
    that can differ from the ERC-4626 accounting asset.  This is raised when the
    selected asset (the ``--deposit-asset`` override, or the native USDC default)
    is not on that whitelist, so the operator sees both the vault's accepted
    assets and the asset that was attempted.
    """

    def __init__(
        self,
        reason: str,
        *,
        selected_asset: str | None = None,
        accepted_assets: list[tuple[str, str]] | None = None,
    ):
        super().__init__(reason)
        #: Address of the asset we tried to deposit, when known.
        self.selected_asset = selected_asset
        #: ``(symbol, address)`` tuples the vault accepts.
        self.accepted_assets = accepted_assets or []


def resolve_multi_asset_deposit_asset(
    deposit_manager,
    chain_id: int,
    override: str | None = None,
) -> str | None:
    """Resolve and validate the accepted input asset for a multi-asset vault.

    Single-asset vaults need no explicit selection and return ``None``.  For a
    multi-asset vault the asset is the ``--deposit-asset`` override, or native
    USDC on the vault's own chain by default.

    :raise IncompatibleDepositAsset:
        When the vault is multi-asset and the selected asset is not on its
        accepted-asset whitelist.  The exception carries both the vault's
        accepted assets and the asset that was attempted.
    """

    fetch_accepted = getattr(deposit_manager, "fetch_accepted_assets", None)
    if fetch_accepted is None:
        # Single-asset ERC-4626 vault: the reserve asset is deposited directly.
        return None

    selected = override or USDC_NATIVE_TOKEN.get(chain_id)
    accepted_pairs = [(token.symbol, token.address) for token in fetch_accepted()]
    accepted_addresses = {address.lower() for _symbol, address in accepted_pairs}
    if selected is None or selected.lower() not in accepted_addresses:
        supported = ", ".join(
            f"{symbol} ({address})" for symbol, address in accepted_pairs
        )
        source = "--deposit-asset override" if override else "native USDC default"
        raise IncompatibleDepositAsset(
            f"Selected deposit asset {selected or '<none>'} ({source}) is not "
            f"accepted by this vault. Vault accepts: {supported or '<none>'}. "
            f"Set --deposit-asset to one of the accepted assets.",
            selected_asset=selected,
            accepted_assets=accepted_pairs,
        )
    return selected


def reconcile_vault_redemption_amount(
    planned_amount: Decimal,
    onchain_balance: Decimal,
    *,
    epsilon: float,
) -> Decimal:
    """Cap a redemption to a tolerably smaller on-chain share balance."""

    assert planned_amount > 0, f"Planned redemption must be positive, got {planned_amount}"
    assert onchain_balance >= 0, f"On-chain share balance cannot be negative, got {onchain_balance}"

    if onchain_balance >= planned_amount:
        return planned_amount

    relative_shortfall = (planned_amount - onchain_balance) / planned_amount
    if relative_shortfall > Decimal(str(epsilon)):
        raise AssertionError(
            f"Vault share token balance has a large relative shortfall: "
            f"planned {planned_amount}, on-chain {onchain_balance}, "
            f"relative shortfall {relative_shortfall}, epsilon {epsilon}"
        )

    return onchain_balance


def convert_vault_flow_analysis(
    analysis: DepositRedeemEventAnalysis,
    *,
    direction: str,
) -> tuple[Decimal, Decimal, Decimal]:
    """Convert a manager receipt analysis to executor trade quantities."""

    executed_reserve = analysis.denomination_amount
    if direction == "deposit":
        executed_amount = analysis.share_count
    else:
        executed_amount = -analysis.share_count
    price = executed_reserve / analysis.share_count
    assert executed_reserve > 0 and executed_amount != 0 and price > 0
    return executed_reserve, executed_amount, price


def _mark_vault_transaction(
    transaction: BlockchainTransaction,
    *,
    role: str,
    request_ordinal: int | None = None,
) -> None:
    """Persist the role of one transaction in a manager-owned vault flow.

    Function selectors are protocol implementation details: an asynchronous
    request may be called ``requestRedeem``, ``redeemShares`` or something not
    yet known to the executor.  The role is therefore the durable settlement
    identity, while a transaction hash remains execution evidence only.
    """

    if transaction.other is None:
        transaction.other = {}
    transaction.other["vault_transaction_role"] = role
    if request_ordinal is not None:
        transaction.other["vault_request_ordinal"] = request_ordinal


def _get_contract_for_vault_function(
    vault: ERC4626Vault,
    function,
):
    """Get the contract object matching a manager-returned bound function.

    Lagoon transaction construction validates that the supplied contract and
    function target agree.  Managers are allowed to return calls directed at a
    protocol helper rather than the vault itself, so retain known token/vault
    contract objects and construct a minimal address wrapper for other targets.
    """

    address = function.address
    if address.lower() == vault.vault_contract.address.lower():
        return vault.vault_contract
    if address.lower() == vault.denomination_token.contract.address.lower():
        return vault.denomination_token.contract
    if vault.share_token and address.lower() == vault.share_token.contract.address.lower():
        return vault.share_token.contract
    return vault.web3.eth.contract(address=address, abi=[])


def get_async_vault_request_transactions(
    trade: TradeExecution,
    *,
    request_function_count: int,
) -> list[BlockchainTransaction]:
    """Select ordered manager request transactions without selector matching.

    New trades carry a role and ordinal on every signed transaction.  A legacy
    pending trade has neither, so use the rebuilt manager request's function
    count to select its final request calls, then upgrade the state in place.
    """

    request_transactions = [
        tx
        for tx in trade.blockchain_transactions
        if (tx.other or {}).get("vault_transaction_role") == "vault_request"
    ]
    if request_transactions:
        request_transactions.sort(
            key=lambda tx: (tx.other or {}).get("vault_request_ordinal", 0)
        )
        return request_transactions

    assert request_function_count > 0, "Manager request must contain at least one call"
    assert len(trade.blockchain_transactions) >= request_function_count, (
        f"Vault trade #{trade.trade_id} has fewer signed transactions than its "
        f"rebuilt manager request: {len(trade.blockchain_transactions)} < {request_function_count}"
    )
    request_transactions = trade.blockchain_transactions[-request_function_count:]
    for ordinal, transaction in enumerate(request_transactions):
        _mark_vault_transaction(
            transaction,
            role="vault_request",
            request_ordinal=ordinal,
        )
    trade.other_data["vault_request_transaction_indices"] = [
        len(trade.blockchain_transactions) - request_function_count + offset
        for offset in range(request_function_count)
    ]
    trade.other_data["vault_request_tx_count"] = request_function_count
    trade.other_data.setdefault("vault_initial_tx_count", len(trade.blockchain_transactions))
    return request_transactions


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
        epsilon=Decimal(1e-6),
        redeem_epsilon=0.025,
        deposit_asset_override: JSONHexAddress | None = None,
    ):
        super().__init__(
            allowed_intermediary_pairs={},
            reserve_token_address=reserve_token_address,
        )
        self.epsilon = epsilon

        # Accepted input asset for multi-asset vaults (e.g. Upshift), where the
        # ERC-4626 accounting asset differs from the deposit assets and the
        # manager requires an explicit selection. ``None`` means "use the
        # default", which :meth:`deposit_or_redeem` resolves to native USDC on
        # the vault's own chain. Set this to override the default per run.
        self.deposit_asset_override = deposit_asset_override

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
        else:
            token_in = trade.pair.base
            token_out = reserve_asset
            # Sells have a negative planned quantity, but redemption requests
            # take a positive share amount.
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
            reconciled_amount = reconcile_vault_redemption_amount(
                swap_amount,
                onchain_balance,
                epsilon=self.redeem_epsilon,
            )
            if reconciled_amount < swap_amount:
                relative_shortfall = (swap_amount - onchain_balance) / swap_amount
                logger.warning(
                    "Vault trade %s, position %s, share token %s, has a small relative difference in onchain balance: %f, planned quantity: %s, onchain balance: %s, automatically rounding, epsilon is %f",
                    trade.trade_id,
                    position,
                    share_token,
                    relative_shortfall,
                    trade.planned_quantity,
                    onchain_balance,
                    self.redeem_epsilon,
                )
                swap_amount = reconciled_amount
            else:
                logger.info(
                    "Onchain balance covers the planned shares to redeem: planned %s, onchain %s",
                    swap_amount,
                    onchain_balance,
                )

        logger.info(
            "Preparing vault flow %s -> %s, amount %s (%s), slippage tolerance %f",
            token_in.token_symbol,
            token_out.token_symbol,
            swap_amount,
            token_in.convert_to_decimal(swap_amount),
            trade.slippage_tolerance,
        )

        deposit_manager = target_vault.get_deposit_manager()

        # The manager owns protocol-specific amount checks and request calls for
        # both synchronous and asynchronous flows.  In particular, this keeps
        # cSigma's owner-specific immediate-redemption capacity check intact.
        if trade.is_buy():
            deposit_kwargs = dict(
                owner=address,
                amount=swap_amount,
                check_enough_token=False,
            )
            # Multi-asset vaults (e.g. Upshift) accept several deposit assets and
            # require an explicit selection; their manager exposes an
            # ``accepted_asset`` parameter.  Default to native USDC on the vault's
            # own chain, overridable per run via ``deposit_asset_override``.  An
            # asset not on the vault whitelist raises IncompatibleDepositAsset.
            # TODO: exercise the override end-to-end (see the vault-test-trade
            # --deposit-asset TODO); only the USDC default is covered today.
            accepted_asset = resolve_multi_asset_deposit_asset(
                deposit_manager,
                target_vault.chain_id,
                self.deposit_asset_override,
            )
            if accepted_asset is not None:
                deposit_kwargs["accepted_asset"] = HexAddress(accepted_asset)
            request = deposit_manager.create_deposit_request(**deposit_kwargs)
            is_async = not deposit_manager.has_synchronous_deposit()
            direction = "deposit"
        else:
            request = deposit_manager.create_redemption_request(
                owner=address,
                shares=swap_amount,
            )
            is_async = not deposit_manager.has_synchronous_redemption()
            direction = "redeem"

        txs: list[BlockchainTransaction] = []

        # Current eth-defi request classes expose manager lifecycle calls in
        # ``funcs`` and a manager-level approval target.  Preserve the selected
        # asset's approval as a separate prerequisite; a request-provided
        # approval call remains in ``funcs`` and is a request transaction.
        if trade.is_buy():
            get_approval_target = getattr(
                deposit_manager,
                "get_deposit_approval_target",
                None,
            )
            approval_target = (
                get_approval_target()
                if get_approval_target is not None
                else target_vault.vault_address
            )
            approve_call = target_vault.denomination_token.approve(
                approval_target,
                swap_amount,
            )
            approve_tx = tx_builder.sign_transaction(
                contract=target_vault.denomination_token.contract,
                args_bound_func=approve_call,
                gas_limit=500_000,
                asset_deltas=[],
                notes=trade.notes,
            )
            _mark_vault_transaction(approve_tx, role="vault_approval")
            txs.append(approve_tx)

        for ordinal, function in enumerate(request.funcs):
            request_tx = tx_builder.sign_transaction(
                contract=_get_contract_for_vault_function(target_vault, function),
                args_bound_func=function,
                gas_limit=self.vault_interaction_gas_limit,
                asset_deltas=[],
                notes=trade.notes,
            )
            _mark_vault_transaction(
                request_tx,
                role="vault_request",
                request_ordinal=ordinal,
            )
            txs.append(request_tx)

        if not is_async:
            return txs

        # Persist request identity and reconstruction inputs.  The transaction
        # indices and roles survive a re-sign; the hashes do not define identity.
        trade.other_data["vault_async_flow"] = True
        trade.other_data["vault_direction"] = direction
        trade.other_data["vault_owner_address"] = address
        trade.other_data["vault_request_tx_count"] = len(request.funcs)
        trade.other_data["vault_initial_tx_count"] = len(txs)
        trade.other_data["vault_request_transaction_indices"] = list(
            range(len(txs) - len(request.funcs), len(txs))
        )
        if direction == "deposit":
            trade.other_data["vault_raw_amount"] = str(request.raw_amount)
            trade.other_data["vault_deposit_amount"] = str(swap_amount)
            try:
                settles_at = deposit_manager.get_deposit_delay_over(address)
            except Exception as error:
                logger.warning(
                    "Could not estimate vault deposit settlement time for %s: %s",
                    target_vault.vault_address,
                    error,
                )
                settles_at = None
            trade.other_data["vault_settlement_estimated_at"] = (
                settles_at.isoformat() if settles_at else None
            )
        else:
            trade.other_data["vault_raw_amount"] = str(request.raw_shares)
            trade.other_data["vault_redeem_shares"] = str(swap_amount)

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
        receipts: dict[str, dict],
        stop_on_execution_failure=False,
    ):

        vault = get_vault_for_pair(
            web3,
            trade.pair,
            token_cache=self.token_cache,
        )
        logger.info(f"Settling vault trade: #{trade.trade_id} for {vault}")

        # Async vault flow — parse request event and mark as pending settlement
        if trade.other_data.get("vault_async_flow"):
            deposit_manager = vault.get_deposit_manager()
            direction = trade.other_data.get("vault_direction", "deposit" if trade.is_buy() else "redeem")
            owner_address = HexAddress(trade.other_data["vault_owner_address"])

            if direction == "deposit":
                # Reconstruct deposit request using raw_amount (int) —
                # all adapters support this path for deposits. int() accepts
                # both the current string form and the legacy int form.
                deposit_request = deposit_manager.create_deposit_request(
                    owner=owner_address,
                    raw_amount=int(trade.other_data["vault_raw_amount"]),
                )
                request_transactions = get_async_vault_request_transactions(
                    trade,
                    request_function_count=len(deposit_request.funcs),
                )
                parse_request = deposit_request.parse_deposit_transaction
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
                request_transactions = get_async_vault_request_transactions(
                    trade,
                    request_function_count=len(redemption_request.funcs),
                )
                parse_request = redemption_request.parse_redeem_transaction

            request_receipts: list[dict] = []
            for request_transaction in request_transactions:
                try:
                    request_receipt = receipts[HexBytes(request_transaction.tx_hash)]
                except KeyError as e:
                    raise KeyError(
                        f"Could not find request hash: {request_transaction.tx_hash} in {receipts}"
                    ) from e
                request_receipts.append(request_receipt)
                if request_receipt["status"] == 0:
                    ts = get_block_timestamp(web3, request_receipt["blockNumber"])
                    report_failure(ts, state, trade, stop_on_execution_failure)
                    return

            # The final manager request call is the lifecycle timestamp. Do not
            # include a preceding token approval or infer the request by name.
            receipt = request_receipts[-1]
            ts = get_block_timestamp(web3, receipt["blockNumber"])
            tx_hashes = [HexBytes(tx.tx_hash) for tx in request_transactions]
            ticket = parse_request(tx_hashes)
            if direction == "deposit":
                ticket_data = deposit_manager.serialize_deposit_ticket(ticket)
            else:
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

        # New manager-owned requests have an explicit role even for a
        # synchronous flow.  This allows a specialised manager to use a
        # non-standard function name without teaching the global swap selector
        # list about its protocol.  Keep selector discovery for legacy trades.
        request_transactions = [
            tx
            for tx in trade.blockchain_transactions
            if (tx.other or {}).get("vault_transaction_role") == "vault_request"
        ]
        swap_tx = request_transactions[-1] if request_transactions else get_swap_transactions(trade)

        try:
            receipt = receipts[HexBytes(swap_tx.tx_hash)]
        except KeyError as e:
            raise KeyError(f"Could not find hash: {swap_tx.tx_hash} in {receipts}") from e

        ts = get_block_timestamp(web3, receipt["blockNumber"])

        # Synchronous vault flow — analyse the deposit/redeem result
        direction = "deposit" if trade.is_buy() else "redeem"

        if receipt["status"] == 0:
            report_failure(ts, state, trade, stop_on_execution_failure)
            return

        deposit_manager = vault.get_deposit_manager()
        manager_analyser = (
            type(deposit_manager).analyse_deposit
            if direction == "deposit"
            else type(deposit_manager).analyse_redemption
        )
        generic_analyser = (
            ERC4626DepositManager.analyse_deposit
            if direction == "deposit"
            else ERC4626DepositManager.analyse_redemption
        )
        if manager_analyser is generic_analyser:
            # The generic manager requires a ticket to identify a GuardV0
            # wrapper. Synchronous executor trades do not persist one yet.
            # Keep its existing guarded-wrapper analyser until eth-defi exposes
            # ticket-free support for this specific compatibility path.
            try:
                result = analyse_4626_flow_transaction(
                    vault=vault,
                    tx_hash=swap_tx.tx_hash,
                    tx_receipt=receipt,
                    direction=direction,
                    hot_wallet=False,
                )
            except Exception as e:
                raise VaultReceiptAnalysisError(
                    f"Failed to analyse vault tx {swap_tx.tx_hash} ({direction})"
                ) from e
        else:
            try:
                if direction == "deposit":
                    result = deposit_manager.analyse_deposit(swap_tx.tx_hash, None)
                else:
                    result = deposit_manager.analyse_redemption(swap_tx.tx_hash, None)
            except Exception as e:
                raise VaultReceiptAnalysisError(
                    f"Failed to analyse vault tx {swap_tx.tx_hash} ({direction})"
                ) from e

        if isinstance(result, TradeSuccess):

            base_token_details = fetch_erc20_details(
                web3,
                trade.pair.base.checksum_address,
                cache=self.token_cache,
                chain_id=trade.pair.base.chain_id,
            )
            reserve = trade.reserve_currency

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

        elif isinstance(result, DepositRedeemEventAnalysis):
            executed_reserve, executed_amount, price = convert_vault_flow_analysis(
                result,
                direction=direction,
            )
            gas_used = receipt.get("gasUsed", 0)
            gas_price = receipt.get("effectiveGasPrice", 0)

            state.mark_trade_success(
                ts,
                trade,
                executed_price=float(price),
                executed_amount=executed_amount,
                executed_reserve=executed_reserve,
                lp_fees=0,
                native_token_price=0,
                cost_of_gas=float(Decimal(gas_used) * Decimal(gas_price) / Decimal(10**18)),
            )
        elif isinstance(result, DepositRedeemEventFailure):
            raise VaultReceiptAnalysisError(
                f"Vault manager could not analyse successful {direction} {swap_tx.tx_hash}: "
                f"{result.revert_reason}"
            )
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
