"""Perform a test trade on a universe."""
import logging
import datetime
from decimal import Decimal

from web3 import Web3

from eth_defi.compat import native_datetime_utc_now
from eth_defi.provider.anvil import is_anvil, mine
from tradeexecutor.state.identifier import TradingPairIdentifier, TradingPairKind
from tradingstrategy.chain import ChainId
from tradingstrategy.lending import LendingReserveDescription
from tradingstrategy.universe import Universe
from tradingstrategy.pair import HumanReadableTradingPairDescription
from tradingstrategy.exchange import ExchangeUniverse

from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel
from tradeexecutor.state.trade import TradeFlag, TradeStatus
from tradeexecutor.state.types import Percent
from tradeexecutor.statistics.core import update_statistics
from tradeexecutor.statistics.statistics_table import serialise_long_short_stats_as_json_table
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.sync_model import SyncModel
from tradeexecutor.utils.accuracy import sum_decimal
from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair
from eth_defi.compat import native_datetime_utc_now

logger = logging.getLogger(__name__)


def _materialise_bridge_on_anvil(
    web3config,
    routing_model: RoutingModel,
    bridge_pair: TradingPairIdentifier,
    dest_chain_id: int,
    fallback_recipient: str,
    bridge_trade,
):
    """Mint USDC on destination Anvil fork to simulate bridge completion.

    On Anvil forks, CctpBridgeRouting uses skip_attestation=True which marks
    the bridge trade as success after burn confirmation, but USDC does not
    physically arrive on the destination chain. This helper materialises
    the tokens by minting directly on the destination Anvil fork.

    The mint recipient is resolved from the CCTP routing model's
    custody_address_resolver, matching what CctpBridgeRouting.setup_trades()
    uses for the real depositForBurn call. For Lagoon/Safe multichain setups
    this is the per-chain Safe address, not the hot wallet.
    """
    from eth_defi.provider.anvil import fund_erc20_on_anvil
    from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details

    # Resolve mint recipient from CCTP routing config
    pair_config = routing_model.pair_configurator.get_config(
        routing_model.pair_configurator.match_router(bridge_pair)
    )
    cctp_routing = pair_config.routing_model
    if hasattr(cctp_routing, 'custody_address_resolver') and cctp_routing.custody_address_resolver:
        recipient = cctp_routing.custody_address_resolver(dest_chain_id)
    else:
        recipient = fallback_recipient

    dest_web3 = web3config.get_connection(ChainId(dest_chain_id))
    usdc_address = USDC_NATIVE_TOKEN[dest_chain_id]

    # Derive raw amount from executed trade using token decimals.
    # fund_erc20_on_anvil uses anvil_setStorageAt which OVERWRITES the balance,
    # so we must read the existing balance first and add to it.
    dest_usdc = fetch_erc20_details(dest_web3, usdc_address)
    bridge_amount_raw = dest_usdc.convert_to_raw(bridge_trade.executed_reserve)
    existing_balance_raw = dest_usdc.contract.functions.balanceOf(
        Web3.to_checksum_address(recipient)
    ).call()
    total_raw = existing_balance_raw + bridge_amount_raw

    logger.info(
        "Anvil fork: materialising %s USDC on chain %s for %s (existing balance: %s)",
        bridge_trade.executed_reserve, ChainId(dest_chain_id).get_name(), recipient,
        dest_usdc.convert_to_decimals(existing_balance_raw),
    )
    fund_erc20_on_anvil(dest_web3, usdc_address, recipient, total_raw)


def _make_cross_chain_test_trade(
    web3: "Web3",
    web3config,
    execution_model: ExecutionModel,
    pricing_model: "PricingModel",
    sync_model: "SyncModel",
    state: State,
    universe: TradingStrategyUniverse,
    routing_model: RoutingModel,
    routing_state: RoutingState,
    position_manager: PositionManager,
    pair: TradingPairIdentifier,
    bridge_pair: TradingPairIdentifier,
    amount: Decimal,
    max_slippage: float,
    buy_only: bool,
    close_only: bool,
    gas_at_start,
    hot_wallet,
    reserve_currency: str,
    reserve_currency_at_start: float,
    anvil_time_skip_seconds: int = 24 * 3600,
):
    """Cross-chain test trade: bridge in, open position, close position, bridge out.

    Handles satellite-chain pairs by automatically injecting CCTP bridge
    trades before opening and after closing positions.

    On real mainnet, CctpBridgeRouting.settle_trade() handles attestation
    and receiveMessage automatically. On Anvil forks, USDC is materialised
    via fund_erc20_on_anvil() after bridge trades succeed.
    """
    from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing

    assert isinstance(pricing_model, GenericPricing), \
        f"Cross-chain test trades require GenericPricing, got {type(pricing_model)}"
    assert hasattr(routing_model, 'pair_configurator'), \
        f"Cross-chain test trades require GenericRouting with pair_configurator, got {type(routing_model)}"

    # Verify both pairs are routable
    try:
        pricing_model.route(bridge_pair)
        pricing_model.route(pair)
    except Exception as e:
        raise RuntimeError(
            f"Pricing model cannot route cross-chain pairs. "
            f"Bridge pair: {bridge_pair.get_ticker()}, satellite pair: {pair.get_ticker()}"
        ) from e

    assert web3config is not None, "Cross-chain test trades require web3config"

    ts = native_datetime_utc_now()
    chain_name = ChainId(pair.chain_id).get_name()
    notes = "A test trade created with perform-test-trade command line command"
    # Judged from the home connection. We assume a home-chain Anvil fork implies
    # the satellite chain is also a fork (the cross-chain test harness forks both;
    # production forks neither), so this also governs satellite force-settlement.
    on_anvil = is_anvil(web3)
    dest_chain_id = pair.chain_id

    def _settle_satellite_async_trade(trade) -> bool:
        return _resolve_satellite_async_settlement(
            trade=trade,
            on_anvil=on_anvil,
            web3config=web3config,
            dest_chain_id=dest_chain_id,
            chain_name=chain_name,
            state=state,
            execution_model=execution_model,
        )

    if close_only:
        # Close-only: find existing positions and close them
        satellite_position = state.portfolio.get_position_by_trading_pair(pair)
        if satellite_position is None or not satellite_position.is_open():
            raise RuntimeError(
                f"Close-only mode but no open position for {pair.get_ticker()} on {chain_name}."
            )

        bridge_position = state.portfolio.get_bridge_position_for_chain(dest_chain_id)
        if bridge_position is None or not bridge_position.is_open():
            raise RuntimeError(
                f"Close-only mode but no open bridge position for chain {chain_name}."
            )

        # Step 1: Close satellite position
        logger.info("Cross-chain close step 1: closing %s on %s", pair.get_ticker(), chain_name)
        ts = native_datetime_utc_now()
        position_manager = PositionManager(
            ts, universe, state, pricing_model,
            default_slippage_tolerance=max_slippage,
        )
        trades = position_manager.close_position(
            satellite_position, notes=notes, flags={TradeFlag.test_trade},
        )
        execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
        if _settle_satellite_async_trade(trades[0]):
            return
        assert trades[0].is_success(), f"Satellite close failed: {trades[0].get_revert_reason()}"

        # Step 2: Bridge back — sized by get_available_bridge_capital()
        logger.info("Cross-chain close step 2: bridging back from %s", chain_name)
        ts = native_datetime_utc_now()
        position_manager = PositionManager(
            ts, universe, state, pricing_model,
            default_slippage_tolerance=max_slippage,
        )
        bridge_back_trades = position_manager.close_position(
            bridge_position, notes=notes, flags={TradeFlag.test_trade},
        )
        execution_model.execute_trades(ts, state, bridge_back_trades, routing_model, routing_state)
        assert bridge_back_trades[0].is_success(), \
            f"Bridge back failed: {bridge_back_trades[0].get_revert_reason()}"

        if on_anvil:
            home_chain_id = web3config.default_chain_id.value
            _materialise_bridge_on_anvil(
                web3config, routing_model, bridge_pair,
                home_chain_id, hot_wallet.address, bridge_back_trades[0],
            )
            sync_model.sync_treasury(
                native_datetime_utc_now(), state, list(universe.reserve_assets),
            )

    else:
        # Buy flow: bridge in, then open satellite position

        # Step 1: Bridge USDC to satellite chain
        logger.info("Cross-chain step 1: bridge %s USDC to %s via CCTP", amount, chain_name)
        bridge_trades = position_manager.open_spot(
            bridge_pair, float(amount), notes=notes, flags={TradeFlag.test_trade},
        )
        execution_model.execute_trades(ts, state, bridge_trades, routing_model, routing_state)
        bridge_buy_trade = bridge_trades[0]
        assert bridge_buy_trade.is_success(), \
            f"Bridge in failed: {bridge_buy_trade.get_revert_reason()}"

        # Materialise USDC on Anvil fork
        if on_anvil:
            _materialise_bridge_on_anvil(
                web3config, routing_model, bridge_pair,
                dest_chain_id, hot_wallet.address, bridge_buy_trade,
            )

        # Step 2: Open position on satellite chain
        logger.info("Cross-chain step 2: open %s on %s", pair.get_ticker(), chain_name)
        ts = native_datetime_utc_now()
        position_manager = PositionManager(
            ts, universe, state, pricing_model,
            default_slippage_tolerance=max_slippage,
        )
        satellite_trades = position_manager.open_spot(
            pair, float(amount), notes=notes, flags={TradeFlag.test_trade},
        )
        execution_model.execute_trades(ts, state, satellite_trades, routing_model, routing_state)
        satellite_buy_trade = satellite_trades[0]
        # Async ERC-7540 (Lagoon) / Ostium V1.5 deposits settle off-chain in a
        # second stage; the request tx succeeds but the trade stays pending.
        if _settle_satellite_async_trade(satellite_buy_trade):
            return
        assert satellite_buy_trade.is_success(), \
            f"Satellite open failed: {satellite_buy_trade.get_revert_reason()}"

        satellite_position = state.portfolio.get_position_by_id(satellite_buy_trade.position_id)
        bridge_position = state.portfolio.get_position_by_id(bridge_buy_trade.position_id)

        long_short_metrics_latest = serialise_long_short_stats_as_json_table(state, None)
        update_statistics(native_datetime_utc_now(), state.stats, state.portfolio,
                          ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)

        if not buy_only:
            # Sell flow: close satellite, then bridge back

            if on_anvil:
                logger.info("Skipping time forward by %d seconds", anvil_time_skip_seconds)
                dest_web3 = web3config.get_connection(ChainId(dest_chain_id))
                mine(dest_web3, increase_timestamp=anvil_time_skip_seconds)

            # Step 3: Close satellite position
            logger.info("Cross-chain step 3: close %s on %s", pair.get_ticker(), chain_name)
            ts = native_datetime_utc_now()
            position_manager = PositionManager(
                ts, universe, state, pricing_model,
                default_slippage_tolerance=max_slippage,
            )
            close_trades = position_manager.close_position(
                satellite_position, notes=notes, flags={TradeFlag.test_trade},
            )
            execution_model.execute_trades(ts, state, close_trades, routing_model, routing_state)
            if _settle_satellite_async_trade(close_trades[0]):
                return
            assert close_trades[0].is_success(), \
                f"Satellite close failed: {close_trades[0].get_revert_reason()}"

            # Step 4: Bridge back — sized by get_available_bridge_capital()
            logger.info("Cross-chain step 4: bridge back from %s", chain_name)
            ts = native_datetime_utc_now()
            position_manager = PositionManager(
                ts, universe, state, pricing_model,
                default_slippage_tolerance=max_slippage,
            )
            bridge_back_trades = position_manager.close_position(
                bridge_position, notes=notes, flags={TradeFlag.test_trade},
            )
            execution_model.execute_trades(
                ts, state, bridge_back_trades, routing_model, routing_state,
            )
            assert bridge_back_trades[0].is_success(), \
                f"Bridge back failed: {bridge_back_trades[0].get_revert_reason()}"

            if on_anvil:
                home_chain_id = web3config.default_chain_id.value
                _materialise_bridge_on_anvil(
                    web3config, routing_model, bridge_pair,
                    home_chain_id, hot_wallet.address, bridge_back_trades[0],
                )
                sync_model.sync_treasury(
                    native_datetime_utc_now(), state, list(universe.reserve_assets),
                )

            long_short_metrics_latest = serialise_long_short_stats_as_json_table(state, None)
            update_statistics(native_datetime_utc_now(), state.stats, state.portfolio,
                              ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)

    # Final report
    gas_at_end = hot_wallet.get_native_currency_balance(web3)
    reserve_currency_at_end = state.portfolio.get_default_reserve_position().get_value()

    logger.info("Cross-chain test trade report")
    logger.info("  Chain: %s", chain_name)
    logger.info("  Gas spent: %s", gas_at_start - gas_at_end)
    logger.info("  Trades done: %d", len(list(state.portfolio.get_all_trades())))
    logger.info("  Reserves: %s %s (was %s)", reserve_currency_at_end, reserve_currency, reserve_currency_at_start)


def _force_vault_settlement_and_resolve(web3, state, trade, execution_model, web3config=None):
    """Force settlement on Anvil for an async vault test trade and resolve it.

    Uses protocol-specific settlement forcing (e.g. tryNewSettlement for Ostium V1.5,
    force_lagoon_settle for ERC-7540) then runs the generic settlement retry.

    :param web3:
        Web3 connection for the chain the vault lives on. For cross-chain
        satellite vaults this must be the destination (satellite) chain
        connection, not the home chain — otherwise the operator-impersonation
        force-settle is sent to the wrong chain.

    :param web3config:
        Multichain web3 config. When given it is forwarded to the settlement
        resolver so the claim transaction is signed and broadcast on the
        vault's own chain (chain-aware claiming for satellite vaults). Without
        it the resolver falls back to the execution model's default connection,
        which is only correct for home-chain vaults.
    """
    from eth_defi.erc_4626.vault_protocol.gains.vault import OstiumVault, OstiumVersion
    from eth_defi.erc_4626.vault_protocol.gains.testing import force_ostium_v15_settlement
    from tradeexecutor.ethereum.vault.vault_routing import get_vault_for_pair
    from tradeexecutor.ethereum.vault.settlement_retry import check_and_resolve_vault_settlements

    # This function impersonates the vault operator and only makes sense against
    # an Anvil fork — on a real chain nobody can force a settlement. Enforce the
    # test-only invariant here rather than relying on every caller's is_anvil()
    # guard, so the dev-account gas payer below can never reach a real chain.
    assert is_anvil(web3), "_force_vault_settlement_and_resolve() is Anvil-only (forces operator settlement)"

    vault = get_vault_for_pair(web3, trade.pair)
    owner = trade.other_data.get("vault_owner_address")

    # Ostium's tryNewSettlement() is permissionless and is broadcast with
    # node-side signing (eth_sendTransaction {"from": ...}), so its gas payer must
    # be an account the node can sign for. The vault owner is the executor's hot
    # wallet — a random key whose local signer middleware lives only on the
    # executor's own web3, not on the Anvil node — so paying from it fails with
    # "No Signer available". On Anvil we therefore pay tryNewSettlement() gas from
    # a node-unlocked dev account instead. (The Lagoon branch below keeps `owner`:
    # its settlement is permissioned and must come from the actual asset manager.)
    settlement_caller = owner
    dev_accounts = web3.eth.accounts
    if dev_accounts:
        settlement_caller = dev_accounts[0]

    # Protocol-specific settlement forcing
    if isinstance(vault, OstiumVault) and vault.version == OstiumVersion.v1_5:
        # May need multiple settlements for withdrawal
        direction = trade.other_data.get("vault_direction", "deposit")
        settlements_needed = 1
        if direction == "redeem":
            withdraw_target = vault.vault_contract.functions.targetSettlementId(False).call()
            last_id = vault.vault_contract.functions.lastSettlementId().call()
            settlements_needed = max(withdraw_target - last_id, 1)
        for _ in range(settlements_needed):
            force_ostium_v15_settlement(vault, settlement_caller)
    else:
        # ERC-7540 (Lagoon) — use force_lagoon_settle if available. Lagoon
        # settlement is permissioned, so it must be sent from the vault manager
        # (the owner for test trades), not the dev account used for Ostium above.
        from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
        if isinstance(vault, LagoonVault):
            from eth_defi.erc_4626.vault_protocol.lagoon.testing import force_lagoon_settle
            force_lagoon_settle(vault, owner)

    # Now run the generic settlement retry. Pass web3config so the claim is
    # broadcast on the vault's own chain for cross-chain satellite vaults.
    resolved = check_and_resolve_vault_settlements(
        state=state,
        execution_model=execution_model,
        web3config=web3config,
    )
    if resolved:
        logger.info("Test trade vault settlement resolved: %d trade(s)", len(resolved))
    else:
        logger.warning("Test trade vault settlement NOT resolved after forcing — may need manual intervention")


def _resolve_satellite_async_settlement(
    *,
    trade,
    on_anvil: bool,
    web3config,
    dest_chain_id: int,
    chain_name: str,
    state: State,
    execution_model,
) -> bool:
    """Resolve an async satellite vault trade left in ``vault_settlement_pending``.

    Satellite-chain vaults can be asynchronous (ERC-7540 Lagoon, Ostium V1.5):
    the request transaction (``requestDeposit`` / ``requestRedeem``) confirms,
    but the trade stays in ``vault_settlement_pending`` until the vault operator
    settles the queue off-chain. The cross-chain test-trade flow must not treat
    that as a hard failure — doing so is the production crash this guards against
    (``AssertionError: Satellite open failed: None``, where the revert reason is
    ``None`` precisely because nothing reverted).

    This mirrors the single-chain handling in :func:`make_test_trade`:

    - **On Anvil** we play the vault operator and force-settle on the
      *destination* chain (``web3config.get_connection(dest_chain_id)``), then
      resolve the claim, so the test trade completes its whole cycle in one run.
      Forcing the settlement on the home-chain connection would send the
      operator transaction to the wrong chain.
    - **On a real chain** nobody can make an off-chain operator settle on
      demand, so we report that settlement is in flight and tell the caller to
      stop. The ``start`` daemon or a re-run of ``perform-test-trade`` claims it
      once the operator has settled.

    :param trade:
        The satellite vault trade just executed (deposit or redeem).

    :param on_anvil:
        Whether the home chain connection is an Anvil fork (test context).

    :param dest_chain_id:
        Chain id of the satellite vault — the chain the settlement happens on.

    :return:
        ``True`` if the caller should stop because settlement is still pending
        off-chain (real chain). ``False`` to continue: either the trade is
        synchronous (already ``success``/failed) or it was force-settled and
        resolved on Anvil.
    """
    if trade.get_status() != TradeStatus.vault_settlement_pending:
        return False

    if on_anvil:
        logger.info(
            "Satellite trade #%d is vault_settlement_pending on Anvil, forcing settlement on chain %s...",
            trade.trade_id, dest_chain_id,
        )
        dest_web3 = web3config.get_connection(ChainId(dest_chain_id))
        _force_vault_settlement_and_resolve(
            dest_web3, state, trade, execution_model, web3config=web3config,
        )
        # Surface a clear error if forcing did not resolve the queue, instead of
        # falling through to the caller's is_success() assertion (which would
        # report the misleading "Satellite open failed: None" this fix removes).
        assert trade.get_status() != TradeStatus.vault_settlement_pending, (
            f"Forced settlement on Anvil did not resolve satellite trade #{trade.trade_id} "
            f"on chain {dest_chain_id}; it is still vault_settlement_pending. Check that the "
            f"test vault operator/asset manager can settle the queue (vault_owner_address)."
        )
        return False

    logger.info(
        "Satellite trade #%d is vault_settlement_pending — async (ERC-7540/Ostium) "
        "settlement happens off-chain on %s. Re-run perform-test-trade after settlement "
        "to complete the cycle.",
        trade.trade_id, chain_name,
    )
    return True


def _resolve_home_chain_async_settlement(
    *,
    trade,
    web3: Web3,
    state: State,
    execution_model,
) -> bool:
    """Resolve a home-chain async vault trade left in ``vault_settlement_pending``.

    Home-chain vaults can be asynchronous (Ostium V1.5, ERC-7540 Lagoon): the
    request transaction (``requestDeposit`` / ``requestWithdraw``) confirms, but
    the trade stays in ``vault_settlement_pending`` until the vault operator
    settles the queue off-chain. ``make_test_trade()`` must not treat that as a
    hard failure — both the buy (open) and sell (close) paths run through this
    helper before their ``is_success()`` assertions.

    Skipping it on the close path was the production crash this guards against:
    ``trade-ui`` / ``perform-test-trade`` closing an Ostium position raised
    ``AssertionError: Test sell failed`` even though the ``requestWithdraw``
    request had confirmed and nothing reverted (``get_revert_reason()`` is
    ``None`` precisely because the redeem request succeeded). The buy path had
    this handling inline; the close path did not, and the two diverged.
    Centralising both paths here keeps them symmetric.

    This is the single-chain sibling of
    :func:`_resolve_satellite_async_settlement`:

    - **On Anvil** we play the vault operator, force-settle and resolve the
      claim in the same run, so the test trade completes its whole cycle.
    - **On a real chain** nobody can make an off-chain operator settle on
      demand, so we report that settlement is in flight and tell the caller to
      stop. The ``start`` daemon or a re-run of ``perform-test-trade`` claims it
      once the operator has settled.

    :param trade:
        The vault trade just executed (deposit or redeem).

    :param web3:
        Web3 connection for the vault's (home) chain.

    :return:
        ``True`` if the caller should stop because settlement is still pending
        off-chain (real chain). ``False`` to continue: either the trade is
        synchronous (already ``success``/failed) or it was force-settled and
        resolved on Anvil.
    """
    if trade.get_status() != TradeStatus.vault_settlement_pending:
        return False

    if is_anvil(web3):
        logger.info(
            "Test trade #%d is vault_settlement_pending on Anvil, forcing settlement...",
            trade.trade_id,
        )
        _force_vault_settlement_and_resolve(web3, state, trade, execution_model)
        # Surface a clear error if forcing did not resolve the queue, instead of
        # falling through to the caller's is_success() assertion (which would
        # report a misleading "Test buy/sell failed" with no revert reason).
        assert trade.get_status() != TradeStatus.vault_settlement_pending, (
            f"Forced settlement on Anvil did not resolve test trade #{trade.trade_id}; "
            f"it is still vault_settlement_pending. Check that the test vault "
            f"operator/keeper can settle the queue (vault_owner_address)."
        )
        return False

    logger.info(
        "Test trade #%d is vault_settlement_pending — async (ERC-7540/Ostium) "
        "settlement happens off-chain. Re-run perform-test-trade after settlement "
        "to complete the cycle.",
        trade.trade_id,
    )
    return True


def make_test_trade(
    web3: Web3,
    execution_model: ExecutionModel,
    pricing_model: PricingModel,
    sync_model: SyncModel,
    state: State,
    universe: TradingStrategyUniverse,
    routing_model: RoutingModel,
    routing_state: RoutingState,
    max_slippage: Percent,
    amount=Decimal("1.0"),
    pair: TradingPairIdentifier | HumanReadableTradingPairDescription | None = None,
    lending_reserve_description: LendingReserveDescription | None = None,
    buy_only: bool = False,
    close_only: bool = False,
    test_short: bool = True,
    anvil_time_skip_seconds: int = 24*3600,
    web3config=None,
):
    """Perform a test trade.

    Buy and sell 1 token worth for 1 USD to check that
    our trade routing works.

    If the pair can be shorted, open and close short position for 1 USD.

    :param buy_only:
        Only open the position, do not close it.

    :param close_only:
        Only close an existing position. Raises if no open position exists.
    """
    assert not (buy_only and close_only), "Cannot set both buy_only and close_only"

    assert isinstance(sync_model, SyncModel)
    assert isinstance(universe, TradingStrategyUniverse)
    assert type(max_slippage) == float
    assert max_slippage > 0, f"max_slippage not set"

    ts = native_datetime_utc_now()

    # Sync nonce for the hot wallet
    logger.info("make_test_trade() at %s", ts)
    execution_model.initialize()

    data_universe: Universe = universe.data_universe

    reserve_asset = universe.get_reserve_asset()

    is_cross_chain = False

    if pair:
        assert not lending_reserve_description

        if not isinstance(pair, TradingPairIdentifier):
            # Resolve human description of the pair
            if data_universe.exchanges:
                exchange_universe = ExchangeUniverse.from_collection(data_universe.exchanges)
            elif data_universe.exchange_universe:
                exchange_universe = data_universe.exchange_universe
            else:
                raise RuntimeError("You need to provide the exchange_universe when creating the universe")

            if len(pair) == 2:
                # By chain id + address
                raw_pair = data_universe.pairs.get_pair_by_smart_contract(pair[1])
            else:
                # By chain id + exchange + base + quote + optional fee
                raw_pair = data_universe.pairs.get_pair(*pair, exchange_universe=exchange_universe)

            pair = translate_trading_pair(raw_pair)

        # Detect if this pair is on a satellite chain (cross-chain trade).
        # Only attempt cross-chain when web3config is explicitly provided
        # (callers managing bridging themselves pass web3config=None).
        # Also skip when the default chain is a test chain (Anvil fork)
        # because the fork chain_id won't match real pair chain_ids.
        from tradeexecutor.ethereum.web3config import TEST_CHAIN_IDS
        is_cross_chain = False
        if web3config is not None and web3config.default_chain_id not in TEST_CHAIN_IDS:
            home_chain_id = web3config.default_chain_id.value
            is_cross_chain = (
                not pair.is_cctp_bridge()
                and pair.chain_id != home_chain_id
            )

        if is_cross_chain:
            chain_name = ChainId(pair.chain_id).get_name()
            logger.info("Cross-chain pair detected: %s on %s (home chain: %s)",
                        pair.get_ticker(), chain_name, ChainId(home_chain_id).get_name())
        else:
            logger.info("Getting price for pair %s using %s", pair, pricing_model)
            # Get estimated price for the asset we are going to buy
            assumed_price_structure = pricing_model.get_buy_price(
                ts,
                pair,
                amount,
            )

            logger.info(
                "Making a test trade on pair: %s, for %f %s price is %f %s/%s",
                pair,
                amount,
                reserve_asset.token_symbol,
                assumed_price_structure.mid_price,
                pair.base.token_symbol,
                reserve_asset.token_symbol,
            )

    elif lending_reserve_description:
        # Convert description to TradingPairIdentifier used in PositionManager
        assert type(lending_reserve_description) in (tuple, list), f"lending_reserve_description must be a tuple, got {type(lending_reserve_description)}"
        credit_pair = pair = lending_reserve = universe.get_lending_reserve_by_human_description(lending_reserve_description)

    logger.info("Sync model is %s", sync_model)
    logger.info("Trading university reserve asset is %s", universe.get_reserve_asset())

    # Sync any incoming stablecoin transfers
    # that have not been synced yet
    balance_updates = sync_model.sync_treasury(
        ts,
        state,
        list(universe.reserve_assets),
        post_valuation=True,
    )

    logger.info("sync_treasury() received balance update events: %s", balance_updates)

    if sync_model.has_position_sync():
        balance_updates = sync_model.sync_positions(
            ts,
            state,
            universe,
            pricing_model,
        )
        logger.info("sync_positions(): received balance update events: %s", balance_updates)

    vault_address = sync_model.get_key_address()
    hot_wallet = sync_model.get_hot_wallet()
    gas_at_start = hot_wallet.get_native_currency_balance(web3)

    logger.info("Account data before test trade")
    logger.info("  Vault address: %s", vault_address)
    logger.info("  Hot wallet address: %s", hot_wallet.address)
    logger.info("  Hot wallet balance: %s", gas_at_start)

    if isinstance(sync_model, EnzymeVaultSyncModel):
        vault = sync_model.vault
        logger.info("  Comptroller address: %s", vault.comptroller.address)
        logger.info("  Vault owner: %s", vault.vault.functions.getOwner().call())
        sync_model.check_ownership()

    if len(state.portfolio.reserves) == 0:
        raise RuntimeError("No reserves detected for the strategy. Does your wallet/vault have USDC deposited for trading?")

    reserve_currency = state.portfolio.get_default_reserve_position().asset.token_symbol
    reserve_currency_at_start = state.portfolio.get_default_reserve_position().get_value()

    logger.info("  Reserve currency balance: %s %s", reserve_currency_at_start, reserve_currency)

    assert reserve_currency_at_start > 0, f"No deposits available to trade. Vault at {vault_address}"

    # Create PositionManager helper class
    # that helps open and close positions
    position_manager = PositionManager(
        ts,
        universe,
        state,
        pricing_model,
        default_slippage_tolerance=max_slippage,
    )

    # Cross-chain dispatch: if the target pair is on a satellite chain,
    # delegate to the cross-chain helper that handles bridge in/out
    if is_cross_chain:
        from tradeexecutor.ethereum.cctp.planner import _find_bridge_pair

        all_pairs = list(universe.iterate_pairs())
        bridge_pair = _find_bridge_pair(all_pairs, pair.chain_id)
        if bridge_pair is None:
            raise RuntimeError(
                f"Cross-chain pair detected on {ChainId(pair.chain_id).get_name()} but no CCTP bridge pair found. "
                f"Ensure auto_generate_cctp_bridges=True in your strategy."
            )

        return _make_cross_chain_test_trade(
            web3=web3,
            web3config=web3config,
            execution_model=execution_model,
            pricing_model=pricing_model,
            sync_model=sync_model,
            state=state,
            universe=universe,
            routing_model=routing_model,
            routing_state=routing_state,
            position_manager=position_manager,
            pair=pair,
            bridge_pair=bridge_pair,
            amount=amount,
            max_slippage=max_slippage,
            buy_only=buy_only,
            close_only=close_only,
            gas_at_start=gas_at_start,
            hot_wallet=hot_wallet,
            reserve_currency=reserve_currency,
            reserve_currency_at_start=reserve_currency_at_start,
            anvil_time_skip_seconds=anvil_time_skip_seconds,
        )

    # The message left on the test positions and trades
    notes = "A test trade created with perform-test-trade command line command"

    # Open the test position only if there isn't position already open
    # on the previous run

    buy_trade = open_short_trade = close_short_trade = open_credit_supply_trade = close_credit_supply_trade = None

    position = state.portfolio.get_position_by_trading_pair(pair)

    if close_only:
        if position is None or not position.is_open():
            raise RuntimeError(
                f"Close-only mode selected but no open position exists for {pair}. "
                f"Open a position first with 'open only' mode."
            )
        logger.info("Close-only mode: skipping buy, proceeding to close position %s", position)

    if position is None and not close_only:
        # Create trades to open the position
        if lending_reserve_description:
            assert lending_reserve
            trades = position_manager.open_credit_supply_position_for_reserves(
                lending_reserve_identifier=lending_reserve,
                amount=float(amount),
                notes=notes,
                flags={TradeFlag.test_trade},
            )

            open_credit_supply_trade = trades[0]
        else:
            trades = position_manager.open_spot(
                pair,
                float(amount),
                notes=notes,
                flags={TradeFlag.test_trade},
            )

        trade = trades[0]
        buy_trade = trade

        # Compose the trades as approve() + swapTokenExact(),
        # broadcast them to the blockchain network and
        # wait for the confirmation
        execution_model.execute_trades(
            ts,
            state,
            trades,
            routing_model,
            routing_state,
        )

        position_id = trade.position_id
        position = state.portfolio.get_position_by_id(position_id)

        assert trade.is_test()
        assert position.is_test()

        # For async vaults (Ostium V1.5, ERC-7540), the open trade enters
        # vault_settlement_pending after execute. On Anvil we force settlement
        # and resolve; on a real chain we report the status and stop.
        if _resolve_home_chain_async_settlement(
            trade=trade, web3=web3, state=state, execution_model=execution_model,
        ):
            return

        if not trade.is_success() or not position.is_open():
            # Alot of diagnostics to debug Arbitrum / WBTC issues
            trades = sum_decimal([t.get_position_quantity() for t in position.trades.values() if t.is_success()])
            direct_balance_updates = position.get_base_token_balance_update_quantity()

            logger.error("Trade quantity: %s, direct balance updates: %s", trades, direct_balance_updates)

            logger.error("Test buy failed: %s", trade)
            logger.error("Tx hash: %s", trade.blockchain_transactions[-1].tx_hash)
            logger.error("Revert reason: %s", trade.blockchain_transactions[-1].revert_reason)
            logger.error("Trade dump:\n%s", trade.get_debug_dump())
            logger.error("Position dump:\n%s", position.get_debug_dump())

        if not trade.is_success():
            raise AssertionError(f"Test buy failed: {trade}, {trade.get_revert_reason()}")

        if not position.is_open():
            raise AssertionError("Test buy succeed, but the position was not opened\n"
                                 "Check for dust corrections.")

        long_short_metrics_latest = serialise_long_short_stats_as_json_table(
            state, None
        )
        
        update_statistics(native_datetime_utc_now(), state.stats, state.portfolio, ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)
    else:
        logger.info("Position %s is already open. No need to open it again.", position)

    logger.info("Position %s is open. Now closing the position.", position)

    if not buy_only:

        logger.info("Position %s is open. Now closing the position.", position)

        if is_anvil(web3):
            # Skip time forward to bypass any potential
            # minimum holding period requirements like with IPOR vaults
            logger.info("Skipping time forward by %d seconds to bypass any minimum holding period requirements", anvil_time_skip_seconds)
            mine(web3, increase_timestamp=anvil_time_skip_seconds)

        # Recreate the position manager for the new timestamp,
        # as time has passed
        ts = native_datetime_utc_now()
        position_manager = PositionManager(
            ts,
            universe,
            state,
            pricing_model,
            default_slippage_tolerance=max_slippage,
        )

        if lending_reserve_description:
            trades = position_manager.close_credit_supply_position(
                position,
                notes=notes,
                flags={TradeFlag.test_trade},
            )
            close_credit_supply_trade = trades[0]
        else:
            trades = position_manager.close_position(
                position,
                notes=notes,
                flags={TradeFlag.test_trade},
            )
        assert len(trades) == 1
        sell_trade = trades[0]

        execution_model.execute_trades(
                ts,
                state,
                [sell_trade],
                routing_model,
                routing_state,
            )

        assert sell_trade.is_test()

        # For async vaults (Ostium V1.5, ERC-7540), closing a position is a
        # requestRedeem()/requestWithdraw() that confirms on-chain but leaves the
        # trade in vault_settlement_pending until the vault operator settles the
        # queue off-chain. This is the symmetric sibling of the open-path
        # handling above: on Anvil we force-settle, on a real chain we report
        # that settlement is in flight and stop. Without it the close path falls
        # through to the is_success() assertion below and crashes with a
        # misleading "Test sell failed" even though the request succeeded and
        # nothing reverted (the production trade-ui Ostium close crash).
        if _resolve_home_chain_async_settlement(
            trade=sell_trade, web3=web3, state=state, execution_model=execution_model,
        ):
            return

        if not sell_trade.is_success():
            logger.error("Test sell failed: %s", sell_trade)
            logger.error("Trade dump:\n%s", sell_trade.get_debug_dump())
            raise AssertionError("Test sell failed")

        long_short_metrics_latest = serialise_long_short_stats_as_json_table(
            state, None
        )
        
        update_statistics(native_datetime_utc_now(), state.stats, state.portfolio, ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)

    else:
        sell_trade = None

    if pair and universe.has_any_lending_data() and universe.can_open_short(native_datetime_utc_now(), pair) and test_short:
        short_pair = universe.get_shorting_pair(pair)
        position = state.portfolio.get_position_by_trading_pair(short_pair)

        if position is None:

            # Recreate the position manager for the new timestamp,
            # as time has passed
            ts = native_datetime_utc_now()
            position_manager = PositionManager(
                ts,
                universe,
                state,
                pricing_model,
                default_slippage_tolerance=max_slippage,
            )

            # Create trades to open the position
            trades = position_manager.open_short(
                pair,
                float(amount),
                notes=notes,
                leverage=2,
                flags={TradeFlag.test_trade},
            )

            trade = trades[0]
            open_short_trade = trade

            # Compose the trades as approve() + swapTokenExact(),
            # broadcast them to the blockchain network and
            # wait for the confirmation
            execution_model.execute_trades(
                ts,
                state,
                trades,
                routing_model,
                routing_state,
            )

            position_id = trade.position_id
            position = state.portfolio.get_position_by_id(position_id)

            assert position.is_test()
            assert open_short_trade.is_test()

            if not trade.is_success() or not position.is_open():
                # Alot of diagnostics to debug Arbitrum / WBTC issues
                trades = sum_decimal([t.get_position_quantity() for t in position.trades.values() if t.is_success()])
                direct_balance_updates = position.get_base_token_balance_update_quantity()

                logger.error("Trade quantity: %s, direct balance updates: %s", trades, direct_balance_updates)

                logger.error("Test open short failed: %s", trade)
                logger.error("Tx hash: %s", trade.blockchain_transactions[-1].tx_hash)
                logger.error("Revert reason: %s", trade.blockchain_transactions[-1].revert_reason)
                logger.error("Trade dump:\n%s", trade.get_debug_dump())
                logger.error("Position dump:\n%s", position.get_debug_dump())

            if not trade.is_success():
                raise AssertionError("Test buy failed.")

            if not position.is_open():
                raise AssertionError("Test buy succeed, but the position was not opened\n"
                                     "Check for dust corrections.")

            long_short_metrics_latest = serialise_long_short_stats_as_json_table(
                state, None
            )

            update_statistics(native_datetime_utc_now(), state.stats, state.portfolio, ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)

        # Close the short

        # Recreate the position manager for the new timestamp,
        # as time has passed
        ts = native_datetime_utc_now()
        position_manager = PositionManager(
            ts,
            universe,
            state,
            pricing_model,
            default_slippage_tolerance=max_slippage,
        )

        # Create trades to open the position
        trades = position_manager.close_short(
            position,
            flags={TradeFlag.test_trade},
        )

        trade = trades[0]
        close_short_trade = trade

        assert close_short_trade.is_test()

        # Compose the trades as approve() + swapTokenExact(),
        # broadcast them to the blockchain network and
        # wait for the confirmation
        execution_model.execute_trades(
            ts,
            state,
            trades,
            routing_model,
            routing_state,
        )

        position_id = trade.position_id
        position = state.portfolio.get_position_by_id(position_id)

        if not trade.is_success() or position.is_open():
            # Alot of diagnostics to debug Arbitrum / WBTC issues
            trades = sum_decimal([t.get_position_quantity() for t in position.trades.values() if t.is_success()])
            direct_balance_updates = position.get_base_token_balance_update_quantity()

            logger.error("Trade quantity: %s, direct balance updates: %s", trades, direct_balance_updates)

            logger.error("Close short failed: %s", trade)
            logger.error("Tx hash: %s", trade.blockchain_transactions[-1].tx_hash)
            logger.error("Revert reason: %s", trade.blockchain_transactions[-1].revert_reason)
            logger.error("Trade dump:\n%s", trade.get_debug_dump())
            logger.error("Position dump:\n%s", position.get_debug_dump())

        if not trade.is_success():
            raise AssertionError(f"Short close failed, trade not marked as success: {trade.get_revert_reason()}")

        if not position.is_closed():
            raise AssertionError("Short close succeed, but the position was not opened\n"
                                 "Check for dust corrections.")

        long_short_metrics_latest = serialise_long_short_stats_as_json_table(
            state, None
        )
        
        update_statistics(native_datetime_utc_now(), state.stats, state.portfolio, ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)

    gas_at_end = hot_wallet.get_native_currency_balance(web3)
    reserve_currency_at_end = state.portfolio.get_default_reserve_position().get_value()

    logger.info("Test trade report")
    logger.info("  Gas spent: %s", gas_at_start - gas_at_end)
    logger.info("  Trades done currently: %d", len(list(state.portfolio.get_all_trades())))
    logger.info("  Reserves currently: %s %s", reserve_currency_at_end, reserve_currency)
    logger.info("  Reserve currency spent: %s %s", reserve_currency_at_start - reserve_currency_at_end, reserve_currency)

    if buy_trade:
        logger.info("  Buy trade price, expected: %s, actual: %s (%s)", buy_trade.planned_price, buy_trade.executed_price, pair.get_ticker())
    if sell_trade:
        logger.info("  Sell trade price, expected: %s, actual: %s (%s)", sell_trade.planned_price, sell_trade.executed_price, pair.get_ticker())
    if open_short_trade:
        logger.info("  Open short, expected: %s, actual: %s (%s)", open_short_trade.planned_price, open_short_trade.executed_price, short_pair.get_ticker())
    if close_short_trade:
        logger.info("  Close short, expected: %s, actual: %s (%s)", close_short_trade.planned_price, close_short_trade.executed_price, short_pair.get_ticker())
    if open_credit_supply_trade:
        logger.info("  Open credit supply, expected: %s, actual: %s (%s)", open_credit_supply_trade.planned_price, open_credit_supply_trade.executed_price, credit_pair.get_ticker())
    if close_credit_supply_trade:
        logger.info("  Close credit supply, expected: %s, actual: %s (%s)", close_credit_supply_trade.planned_price, close_credit_supply_trade.executed_price, credit_pair.get_ticker())
