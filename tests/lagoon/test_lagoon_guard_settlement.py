"""GuardV0 Lagoon settlement coverage on a Base Anvil fork."""
import logging
from decimal import Decimal

import pytest
from eth_typing import HexAddress
from web3 import Web3

from eth_defi.erc_4626.vault_protocol.lagoon.deployment import (
    LagoonAutomatedDeployment,
    LagoonDeploymentParameters,
    deploy_automated_lagoon_vault,
)
from eth_defi.compat import native_datetime_utc_now
from eth_defi.hotwallet import HotWallet
from eth_defi.token import TokenDetails, USDC_NATIVE_TOKEN
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def _request_deposit(
    vault: LagoonAutomatedDeployment,
    token: TokenDetails,
    depositor: HexAddress,
    amount: Decimal,
) -> None:
    """Queue a Lagoon deposit using an investor account."""
    raw_amount = token.convert_to_raw(amount)
    tx_hash = token.approve(vault.vault.address, amount).transact({"from": depositor})
    assert_transaction_success_with_explanation(vault.vault.web3, tx_hash)
    tx_hash = vault.vault.request_deposit(depositor, raw_amount).transact({"from": depositor})
    assert_transaction_success_with_explanation(vault.vault.web3, tx_hash)


@pytest.fixture()
def guarded_lagoon_vault(
    web3: Web3,
    deployer_hot_wallet: HotWallet,
    asset_manager: HotWallet,
    multisig_owners: list[HexAddress],
) -> LagoonAutomatedDeployment:
    """Deploy a stock Lagoon v0.5 vault with a 10 USDC GuardV0 settlement limit."""
    return deploy_automated_lagoon_vault(
        web3=web3,
        deployer=deployer_hot_wallet,
        asset_manager=asset_manager.address,
        parameters=LagoonDeploymentParameters(
            underlying=USDC_NATIVE_TOKEN[web3.eth.chain_id],
            name="Guarded Lagoon",
            symbol="GLG",
        ),
        safe_owners=multisig_owners,
        safe_threshold=2,
        uniswap_v2=None,
        uniswap_v3=None,
        any_asset=True,
        max_settlement_amount=Decimal(10),
    )


def test_lagoon_guard_automatically_settles_flow_below_settlement_limit(
    web3: Web3,
    guarded_lagoon_vault: LagoonAutomatedDeployment,
    base_usdc_token: TokenDetails,
    vault_strategy_universe: TradingStrategyUniverse,
    depositor: HexAddress,
    asset_manager: HotWallet,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Settle a below-cap investor queue automatically and post NAV without an empty settlement.

    1. Initialise the executor treasury for a fresh GuardV0-capped Lagoon vault.
    2. Queue a 9 USDC investor deposit below the 10 USDC Guard limit.
    3. Post NAV and automatically settle the deposit through the guarded module.
    4. Queue another below-cap deposit during the cooldown and verify NAV-only deferral.
    """
    vault = guarded_lagoon_vault.vault
    sync_model = LagoonVaultSyncModel(vault=vault, hot_wallet=asset_manager)
    state = State()
    sync_model.sync_initial(state, reserve_asset=vault_strategy_universe.get_reserve_asset(), reserve_token_price=1.0)

    # 1. Initialise the executor treasury for a fresh GuardV0-capped Lagoon vault.
    assert state.portfolio.get_cash() == 0

    # 2. Queue a 9 USDC investor deposit below the 10 USDC Guard limit.
    _request_deposit(guarded_lagoon_vault, base_usdc_token, depositor, Decimal(9))

    # 3. Post NAV and automatically settle the deposit through the guarded module.
    with caplog.at_level(logging.ERROR):
        events = sync_model.sync_treasury(native_datetime_utc_now(), state, post_valuation=True)
    assert len(events) == 1
    assert base_usdc_token.fetch_balance_of(vault.silo_address) == Decimal(0)
    assert state.portfolio.get_cash() == pytest.approx(9)
    assert vault.get_flow_manager().fetch_pending_deposit(web3.eth.block_number) == Decimal(0)
    safety_config = vault.trading_strategy_module.functions.getLagoonSettlementSafetyConfig(vault.address).call()
    assert safety_config[6] > 0
    assert safety_config[7] == safety_config[6] + 86_400
    assert "direct Safe-governance settlement required" not in caplog.text

    # 4. Queue another below-cap deposit during the cooldown and verify NAV-only deferral.
    _request_deposit(guarded_lagoon_vault, base_usdc_token, depositor, Decimal(1))
    nonce_before = web3.eth.get_transaction_count(asset_manager.address)
    with caplog.at_level(logging.INFO):
        events = sync_model.sync_treasury(native_datetime_utc_now(), state, post_valuation=True)
    assert events == []
    assert web3.eth.get_transaction_count(asset_manager.address) == nonce_before + 1
    assert base_usdc_token.fetch_balance_of(vault.silo_address) == Decimal(1)
    assert vault.trading_strategy_module.functions.getLagoonSettlementSafetyConfig(vault.address).call()[6:] == safety_config[6:]
    assert "The queue will be retried automatically" in caplog.text


def test_lagoon_guard_posts_nav_but_defers_oversized_flow_to_safe(
    web3: Web3,
    guarded_lagoon_vault: LagoonAutomatedDeployment,
    base_usdc_token: TokenDetails,
    vault_strategy_universe: TradingStrategyUniverse,
    depositor: HexAddress,
    asset_manager: HotWallet,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Post new NAV but preserve an oversized gross queue for direct Safe settlement.

    1. Bootstrap and finalise a below-cap deposit so the investor has redeemable shares.
    2. Queue opposing deposit and redemption flows whose gross amount exceeds the Guard cap.
    3. Run the post-valuation treasury sync once more.
    4. Verify it broadcasts NAV only, preserves both queues and emits the Safe-governance error.
    """
    vault = guarded_lagoon_vault.vault
    sync_model = LagoonVaultSyncModel(vault=vault, hot_wallet=asset_manager)
    state = State()
    sync_model.sync_initial(state, reserve_asset=vault_strategy_universe.get_reserve_asset(), reserve_token_price=1.0)

    # 1. Bootstrap and finalise a below-cap deposit so the investor has redeemable shares.
    _request_deposit(guarded_lagoon_vault, base_usdc_token, depositor, Decimal(5))
    assert len(sync_model.sync_treasury(native_datetime_utc_now(), state, post_valuation=True)) == 1
    tx_hash = vault.finalise_deposit(depositor).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    assert vault.share_token.fetch_balance_of(depositor) == pytest.approx(Decimal(5))

    # 2. Queue opposing deposit and redemption flows whose gross amount exceeds the Guard cap.
    _request_deposit(guarded_lagoon_vault, base_usdc_token, depositor, Decimal(9))
    redeem_raw = vault.share_token.convert_to_raw(Decimal(2))
    tx_hash = vault.request_redeem(depositor, redeem_raw).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    safety_before = vault.trading_strategy_module.functions.getLagoonSettlementSafetyConfig(vault.address).call()
    safe_balance_before = base_usdc_token.fetch_balance_of(vault.safe_address)
    vault_balance_before = base_usdc_token.fetch_balance_of(vault.address)
    nonce_before = web3.eth.get_transaction_count(asset_manager.address)

    # 3. Run the post-valuation treasury sync once more.
    cycle = native_datetime_utc_now()
    with caplog.at_level(logging.ERROR):
        events = sync_model.sync_treasury(cycle, state, post_valuation=True)

    # 4. Verify it broadcasts NAV only, preserves both queues and emits the Safe-governance error.
    assert events == []
    assert web3.eth.get_transaction_count(asset_manager.address) == nonce_before + 1
    assert state.sync.treasury.last_cycle_at == cycle
    assert base_usdc_token.fetch_balance_of(vault.silo_address) == Decimal(9)
    assert vault.share_token.fetch_raw_balance_of(vault.silo_address) == redeem_raw
    assert base_usdc_token.fetch_balance_of(vault.safe_address) == safe_balance_before
    assert base_usdc_token.fetch_balance_of(vault.address) == vault_balance_before
    assert vault.trading_strategy_module.functions.getLagoonSettlementSafetyConfig(vault.address).call()[6:] == safety_before[6:]
    assert state.sync.treasury.pending_redemptions == pytest.approx(2)
    assert "direct Safe-governance settlement required" in caplog.text
    assert "gross_flow=" in caplog.text
    assert "(10999999 raw)" in caplog.text
    assert "cap=10" in caplog.text
    assert vault.address in caplog.text
    assert vault.safe_address in caplog.text
