"""CCTP in-transit trade retry on startup.

When the executor process restarts while a CCTP bridge trade is in the
``cctp_in_transit`` state (burn confirmed, receive pending), this module
provides a best-effort function to poll Circle's attestation API and
broadcast ``receiveMessage`` on the destination chain.
"""

import logging
import datetime

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.ethereum.cctp.routing import estimate_receive_message_gas
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus

logger = logging.getLogger(__name__)


def check_and_retry_cctp_in_transit(
    state: State,
    execution_model,
    web3config,
    attestation_timeout: float = 300.0,
) -> list[TradeExecution]:
    """Check for cctp_in_transit trades and attempt to complete them.

    Called during live execution startup to resolve any bridges
    that were interrupted by a process restart.

    The function scans all open positions for trades with
    ``cctp_in_transit`` status.  For each one it:

    1. Reads ``cctp_dest_chain_id`` and ``cctp_burn_tx_hash`` from
       ``trade.other_data``.
    2. Inspects ``trade.blockchain_transactions`` to determine whether a
       ``receiveMessage`` transaction was already attempted.
    3. If no receive tx exists, polls attestation and broadcasts
       ``receiveMessage`` on the destination chain.
    4. If a receive tx exists but was not confirmed, rebroadcasts it.
    5. On success: clears ``cctp_in_transit_at``, reverses
       ``bridge_capital_allocated`` for bridge-back sells, and calls
       ``mark_trade_success``.
    6. On failure: logs a warning and includes the trade in the
       returned unresolved list.

    :param state:
        Current strategy state.

    :param execution_model:
        The execution model (must have ``tx_builder.hot_wallet``).

    :param web3config:
        Web3Config with connections to all chains.

    :param attestation_timeout:
        Maximum seconds to wait for attestation (per trade).

    :return:
        List of trades that were successfully resolved.
    """
    from decimal import Decimal

    from eth_defi.cctp.attestation import fetch_attestation
    from eth_defi.cctp.receive import prepare_receive_message
    from eth_defi.cctp.transfer import _resolve_cctp_domain, get_message_transmitter_v2
    from eth_defi.hotwallet import HotWallet
    from hexbytes import HexBytes
    from tradingstrategy.chain import ChainId

    from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder

    resolved: list[TradeExecution] = []

    # 1. Scan all open positions for cctp_in_transit trades
    in_transit_trades: list[TradeExecution] = []
    for position in state.portfolio.open_positions.values():
        for trade in position.trades.values():
            if trade.get_status() == TradeStatus.cctp_in_transit:
                in_transit_trades.append(trade)

    if not in_transit_trades:
        return resolved

    logger.info(
        "Found %d CCTP in-transit trade(s) on startup, attempting retry",
        len(in_transit_trades),
    )

    for trade in in_transit_trades:
        try:
            # 2. Read metadata from trade.other_data
            dest_chain_id = trade.other_data.get("cctp_dest_chain_id")
            source_chain_id = trade.other_data.get("cctp_source_chain_id")
            burn_tx_hash = trade.other_data.get("cctp_burn_tx_hash")

            if dest_chain_id is None or burn_tx_hash is None:
                logger.warning(
                    "CCTP in-transit trade %s missing metadata, skipping",
                    trade.trade_id,
                )
                continue

            source_domain = _resolve_cctp_domain(source_chain_id)
            if source_domain is None:
                logger.warning(
                    "No CCTP domain for source chain %d, skipping trade %s",
                    source_chain_id, trade.trade_id,
                )
                continue

            dest_web3 = web3config.get_connection(ChainId(dest_chain_id))
            if dest_web3 is None:
                logger.warning(
                    "No web3 connection for dest chain %d, skipping trade %s",
                    dest_chain_id, trade.trade_id,
                )
                continue

            # 3. Inspect blockchain_transactions for existing receive tx
            has_receive_tx = len(trade.blockchain_transactions) > 2
            receive_already_confirmed = False

            if has_receive_tx:
                existing_receive_tx = trade.blockchain_transactions[-1]
                # Check if it was already confirmed on-chain
                if existing_receive_tx.tx_hash:
                    try:
                        receipt = dest_web3.eth.get_transaction_receipt(
                            HexBytes(existing_receive_tx.tx_hash)
                        )
                        if receipt and receipt["status"] == 1:
                            receive_already_confirmed = True
                        elif receipt and receipt["status"] == 0:
                            # Previous receive reverted, need fresh attempt
                            has_receive_tx = False
                    except Exception:
                        # Tx not found — may need rebroadcast
                        pass

            if receive_already_confirmed:
                # Receive already confirmed, just update state
                ts = native_datetime_utc_now()
            elif has_receive_tx:
                # 4. Rebroadcast existing receive tx
                existing_receive_tx = trade.blockchain_transactions[-1]
                try:
                    dest_web3.eth.send_raw_transaction(
                        HexBytes(existing_receive_tx.signed_bytes)
                    )
                    receipt = dest_web3.eth.wait_for_transaction_receipt(
                        HexBytes(existing_receive_tx.tx_hash),
                        timeout=120,
                    )
                    if receipt["status"] != 1:
                        logger.warning(
                            "CCTP receive rebroadcast reverted for trade %s",
                            trade.trade_id,
                        )
                        continue
                except Exception as e:
                    logger.warning(
                        "CCTP receive rebroadcast failed for trade %s: %s",
                        trade.trade_id, e,
                    )
                    continue
                ts = native_datetime_utc_now()
            else:
                # No receive tx yet — poll attestation and broadcast fresh
                try:
                    attestation = fetch_attestation(
                        source_domain=source_domain,
                        transaction_hash=burn_tx_hash,
                        timeout=attestation_timeout,
                    )
                except TimeoutError:
                    logger.warning(
                        "CCTP attestation still pending for trade %s after %.0fs",
                        trade.trade_id, attestation_timeout,
                    )
                    continue

                # 5. Clone hot wallet and broadcast receiveMessage
                dest_wallet = HotWallet(execution_model.tx_builder.hot_wallet.account)
                dest_wallet.sync_nonce(dest_web3)

                message_transmitter = get_message_transmitter_v2(dest_web3)
                receive_fn = prepare_receive_message(
                    dest_web3,
                    attestation.message,
                    attestation.attestation,
                )

                dest_tx_builder = HotWalletTransactionBuilder(dest_web3, dest_wallet)
                receive_tx = dest_tx_builder.sign_transaction(
                    message_transmitter,
                    receive_fn,
                    gas_limit=estimate_receive_message_gas(receive_fn, dest_wallet.address),
                    asset_deltas=[],
                    notes=f"CCTP receiveMessage retry chain {source_chain_id} -> {dest_chain_id}",
                )

                trade.blockchain_transactions.append(receive_tx)

                try:
                    dest_web3.eth.send_raw_transaction(HexBytes(receive_tx.signed_bytes))
                    receive_tx.broadcasted_at = native_datetime_utc_now()
                    receipt = dest_web3.eth.wait_for_transaction_receipt(
                        HexBytes(receive_tx.tx_hash), timeout=120,
                    )
                except Exception as e:
                    logger.warning(
                        "CCTP receiveMessage retry broadcast failed for trade %s: %s",
                        trade.trade_id, e,
                    )
                    continue

                if receipt["status"] != 1:
                    logger.warning(
                        "CCTP receiveMessage retry reverted for trade %s",
                        trade.trade_id,
                    )
                    continue

                ts = native_datetime_utc_now()

            # 6. Success — clear in-transit and mark trade complete
            trade.cctp_in_transit_at = None

            # Reverse bridge_capital_allocated for bridge-back sells
            if trade.is_sell():
                bridge_position = state.portfolio.get_position_by_id(trade.position_id)
                if bridge_position is not None:
                    bridge_position.bridge_capital_allocated -= abs(trade.planned_quantity)

            state.mark_trade_success(
                ts,
                trade,
                executed_price=1.0,
                executed_amount=trade.planned_quantity,
                executed_reserve=trade.planned_reserve,
                lp_fees=0,
                native_token_price=0,
            )

            resolved.append(trade)
            logger.info(
                "CCTP in-transit trade %s resolved successfully",
                trade.trade_id,
            )

        except Exception as e:
            logger.warning(
                "Unexpected error retrying CCTP in-transit trade %s: %s",
                trade.trade_id, e,
                exc_info=True,
            )

    logger.info(
        "CCTP startup retry complete: %d/%d trades resolved",
        len(resolved), len(in_transit_trades),
    )

    return resolved
