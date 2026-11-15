"""Execute a staged Morpho exit before the live loop's account check.

The reviewed block 26402309 is a deliberate lifecycle exception to the shared
midnight block: it reproduces the studied exit-market shortfall. Guard custody
and both real protocol receipts are independently covered in eth-defi and the
public Safe CLI simulation. This test targets the executor's live receipt,
checkpoint, valuation and wallet reconciliation boundaries with a hot wallet.
"""

import datetime
import os
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest
from eth_account import Account
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fund_erc20_on_anvil, make_anvil_custom_rpc_request, mine, set_balance
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.evm_snapshot_fixture import evm_snapshot_revert
from eth_defi.erc_4626.vault_protocol.morpho.vault_v2 import MorphoV2Vault
from eth_defi.vault.base import VaultSpec
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType, ExchangeUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from tradingstrategy.vault import VaultMetadata
from eth_defi.erc_4626.core import ERC4626Feature
from web3 import Web3

from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.lagoon.universe_config import translate_trading_universe_to_lagoon_config
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.ethereum.vault.vault_live_pricing import VaultPricing
from tradeexecutor.ethereum.vault.vault_routing import VaultRouting
from tradeexecutor.ethereum.vault.vault_valuation import VaultValuator
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.state.trade import TradeStatus, TradeType
from tradeexecutor.strategy.account_correction import check_accounts
from tradeexecutor.strategy.reverse_universe import create_universe_from_trading_pair_identifiers
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.vault_protocol import enrich_vault_pair

RPC = os.environ.get("JSON_RPC_ETHEREUM")
BLOCK = 26_402_309
pytestmark = [pytest.mark.skipif(not RPC, reason="JSON_RPC_ETHEREUM required"), pytest.mark.xdist_group("fork:ethereum:morpho-recovery")]


@pytest.fixture
def web3(anvil_fork_pool: AnvilForkPool) -> Iterator[Web3]:
    """Restore the fixed shared fork and deterministic block clock after the test."""
    launch = anvil_fork_pool.get_launch(RPC, BLOCK)
    connection = anvil_fork_pool.get_web3(RPC, BLOCK)
    snapshots = evm_snapshot_revert(launch)
    next(snapshots)
    try:
        make_anvil_custom_rpc_request(connection, "anvil_setBlockTimestampInterval", [1])
        mine(connection, timestamp=connection.eth.get_block(BLOCK)["timestamp"] + 1)
        yield connection
    finally:
        make_anvil_custom_rpc_request(connection, "anvil_removeBlockTimestampInterval", [])
        next(snapshots, None)


def test_morpho_live_execution_finishes_before_account_check(web3: Web3, tmp_path: Path) -> None:
    """Use actual routing, receipt analysis and the persisted settlement resolver.

    1. Fund a test wallet and execute a real 1,001 USDC deposit.
    2. Hold 24 hours and checkpoint the fee-paid pending stage, verifying wallet
       balances, unavailable trading quantity and net pending valuation.
    3. Let EthereumExecution finish immediately, then run the live account check.
    4. Reload the final State and verify one fee, full disposal and two exit calls.
    """
    # 1. Only initial wallet funding is synthetic; vault assets are never injected.
    vault = MorphoV2Vault(web3, VaultSpec(1, "0x36cfe1568461e499391ef0a555300f1ae2da2439"))
    wallet = HotWallet(Account.create())
    set_balance(web3, wallet.address, Web3.to_wei(10, "ether"))
    wallet.sync_nonce(web3)
    quote = AssetIdentifier(1, vault.denomination_token.address, "USDC", vault.denomination_token.decimals)
    base = AssetIdentifier(1, vault.address, "Api3", vault.share_token.decimals)
    pair = TradingPairIdentifier(base, quote, vault.address, "0x" + "00" * 20, fee=0, kind=TradingPairKind.vault, internal_id=1,
        other_data={"token_metadata": VaultMetadata(vault_name="Api3", protocol_name="Morpho", protocol_slug="morpho", features=[ERC4626Feature.morpho_v2_like], deposit_fee=0, withdrawal_fee=0)})
    enrich_vault_pair(pair)
    exchange = Exchange(chain_id=ChainId.ethereum, chain_slug="ethereum", exchange_id=1, exchange_slug="morpho", address=pair.exchange_address, exchange_type=ExchangeType.erc_4626_vault, pair_count=1)
    exchanges = ExchangeUniverse({1: exchange})
    pairs = create_universe_from_trading_pair_identifiers([pair], exchange_universe=exchanges)
    universe = TradingStrategyUniverse(data_universe=Universe(chains={ChainId.ethereum}, time_bucket=TimeBucket.not_applicable, exchange_universe=exchanges, pairs=pairs), reserve_assets=[quote])
    config = translate_trading_universe_to_lagoon_config(
        universe=universe, chain_web3={"ethereum": web3}, asset_manager=wallet.address,
        safe_owners=[wallet.address], safe_threshold=1, safe_salt_nonce=42,
    )["ethereum"]
    assert config.erc_4626_vaults[0].redemption_recovery_max_fee_bps == 1
    state = State()
    state.portfolio.initialise_reserves(quote)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity, reserve.reserve_token_price = Decimal(2000), 1.0
    fund_erc20_on_anvil(web3, vault.denomination_token.address, wallet.address, vault.denomination_token.convert_to_raw(Decimal(2000)))
    execution = EthereumExecution(HotWalletTransactionBuilder(web3, wallet), mainnet_fork=True, confirmation_block_count=0)
    routing = VaultRouting(quote.address)
    routing_state = routing.create_routing_state(universe, execution.get_routing_state_details())
    sync = HotWalletSyncModel(web3, wallet)
    store = JSONFileStore(tmp_path / "state.json")
    execution.set_pre_broadcast_state_sync_callback(lambda: store.sync(state))
    ts = datetime.datetime(2026, 11, 15)
    _, buy, _ = state.create_trade(ts, pair, None, Decimal(1001), 1.0, TradeType.rebalance, quote, 1.0, slippage_tolerance=0.005)
    execution.execute_trades(ts, state, [buy], routing, routing_state)
    assert buy.is_success()

    # 2. A real checkpoint between the two receipts must match custody balances.
    mine(web3, increase_timestamp=86400)
    position = state.portfolio.find_position_for_trade(buy)
    original = position.get_quantity()
    _, sell, _ = state.create_trade(ts, pair, -original, None, 1.0, TradeType.rebalance, quote, 1.0, slippage_tolerance=0.005)
    observed_pending = []
    def checkpoint() -> None:
        store.sync(state)
        if sell.get_status() is TradeStatus.vault_settlement_pending and len(sell.blockchain_transactions) == 1:
            persisted = State.from_json(state.to_json())
            held = dict(position.get_held_assets())[base]
            assert held == vault.share_token.fetch_balance_of(wallet.address)
            assert position.get_available_trading_quantity() == 0
            clean, differences = check_accounts(pairs, [quote], persisted, sync, block_identifier=web3.eth.block_number, check_clean=False)
            assert clean, differences
            valuation = VaultValuator(VaultPricing(web3, owner_address_resolver=lambda pair: wallet.address))
            valuation(ts, position)
            manager = vault.get_deposit_manager()
            ticket = manager.reconstruct_redemption_ticket(sell.other_data)
            assert position.get_value() == pytest.approx(float(manager.fetch_pending_redemption_value(ticket)))
            observed_pending.append(sell.other_data["morpho_penalty_shares"])
    execution.set_pre_broadcast_state_sync_callback(checkpoint)

    # 3. Execute uses the same immediate resolver as the live loop, without a CLI sweep.
    execution.execute_trades(ts, state, [sell], routing, routing_state)
    assert observed_pending and sell.is_success()
    clean, differences = check_accounts(pairs, [quote], state, sync, block_identifier=web3.eth.block_number)
    assert clean, differences

    # 4. Reopening State preserves one paid fee and the original full disposal.
    restored = store.load()
    exit_trade = restored.portfolio.get_trade_by_id(sell.trade_id)
    assert exit_trade.executed_quantity == -original
    assert Decimal(exit_trade.other_data["vault_withdrawal_fee"]) > 0
    assert len(exit_trade.blockchain_transactions) == 2
    assert vault.share_token.fetch_raw_balance_of(wallet.address) == 0
    assert execution.resolve_pending_vault_settlements(state=restored, ts=ts) == []
