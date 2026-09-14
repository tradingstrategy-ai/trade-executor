"""Forked Lagoon NAV integration coverage for a Lighter account position.

The test keeps Lagoon, Safe, ERC-20 reads, treasury reconciliation and NAV
posting real. Only Lighter's external public account observation is mocked,
because a local Anvil fork cannot expose its account state to the sequencer.
"""

import os
from decimal import Decimal

import pytest
from eth_account import Account
from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.settlement_events import fetch_vault_settlement_logs
from eth_defi.erc_4626.vault_protocol.lagoon.config import get_lagoon_chain_config
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import (
    LagoonDeploymentParameters,
    deploy_automated_lagoon_vault,
)
from eth_defi.erc_4626.vault_protocol.lagoon.funding import fund_lagoon_vault
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.evm_snapshot_fixture import evm_snapshot_revert
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_NATIVE_TOKEN, USDC_WHALE, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from pytest_mock import MockerFixture
from web3 import Web3

from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.exchange_account.lighter import (
    NegativeLighterEquityError,
    create_lighter_account_value_func,
    create_lighter_exchange_account_pair,
    create_lighter_vault_valuation_func,
)
from tradeexecutor.exchange_account.pricing import ExchangeAccountPricingModel
from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.exchange_account.valuation import ExchangeAccountValuator
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State

JSON_RPC_ETHEREUM = os.environ.get("JSON_RPC_ETHEREUM")

pytestmark = [
    pytest.mark.skipif(
        not JSON_RPC_ETHEREUM,
        reason="JSON_RPC_ETHEREUM environment variable required",
    ),
    pytest.mark.warm_rpc_test_group,
    pytest.mark.xdist_group("fork:ethereum:midnight"),
]

DEPLOYER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
LIGHTER_ACCOUNT_INDEX = 124
SAFE_USDC = Decimal("5")
LIGHTER_COLLATERAL = Decimal("8")
LIGHTER_UNREALISED_PNL = Decimal("-0.75")
LIGHTER_TOTAL_EQUITY = Decimal("7.25")


@pytest.fixture()
def anvil_ethereum(anvil_fork_pool: AnvilForkPool) -> AnvilLaunch:
    """Reset the shared fixed Ethereum fork around the NAV test."""
    launch = anvil_fork_pool.get_launch(
        JSON_RPC_ETHEREUM,
        ETHEREUM_MIDNIGHT_BLOCK,
        unlocked_addresses=[USDC_WHALE[1]],
    )
    snapshot = evm_snapshot_revert(launch)
    next(snapshot)
    try:
        yield launch
    finally:
        next(snapshot, None)


@pytest.fixture()
def web3_ethereum(anvil_ethereum: AnvilLaunch) -> Web3:
    web3 = create_multi_provider_web3(
        anvil_ethereum.json_rpc_url,
        default_http_timeout=(3, 250.0),
    )
    assert web3.eth.chain_id == 1
    return web3


@pytest.fixture()
def deployer(web3_ethereum: Web3) -> HotWallet:
    account = Account.from_key(DEPLOYER_PRIVATE_KEY)
    web3_ethereum.provider.make_request(
        "anvil_setBalance",
        [account.address, hex(100 * 10**18)],
    )
    return HotWallet(account)


@pytest.mark.timeout(600)
def test_lighter_lagoon_nav_uses_safe_balance_and_total_equity(
    web3_ethereum: Web3,
    deployer: HotWallet,
    mocker: MockerFixture,
) -> None:
    """Post real Lagoon NAV as Safe USDC plus mocked Lighter total equity.

    1. Deploy a real Ethereum Lagoon vault and fund it through the normal
       subscription, valuation, settlement and claim lifecycle.
    2. Create and revalue a Lighter exchange-account position through the
       production public-reader adapter.
    3. Reconcile the Safe and post NAV through the real Lagoon sync model.
    4. Verify the state components, on-chain reserve and posted total assets.
    5. Reject a negative public-equity response before another NAV transaction.
    """
    # 1. Deploy a real Ethereum Lagoon vault and fund it through the normal
    # subscription, valuation, settlement and claim lifecycle.
    usdc = fetch_erc20_details(
        web3_ethereum,
        USDC_NATIVE_TOKEN[1],
        chain_id=1,
    )
    funding_tx = usdc.contract.functions.transfer(
        deployer.address,
        usdc.convert_to_raw(SAFE_USDC + Decimal("2")),
    ).transact({"from": USDC_WHALE[1]})
    assert_transaction_success_with_explanation(web3_ethereum, funding_tx)

    lagoon_chain_config = get_lagoon_chain_config(1)
    deployment = deploy_automated_lagoon_vault(
        web3=web3_ethereum,
        deployer=deployer,
        asset_manager=deployer.address,
        parameters=LagoonDeploymentParameters(
            underlying=USDC_NATIVE_TOKEN[1],
            name="Lighter NAV test",
            symbol="LNV",
        ),
        safe_owners=[deployer.address],
        safe_threshold=1,
        any_asset=True,
        use_forge=True,
        factory_contract=lagoon_chain_config.factory_contract,
        from_the_scratch=lagoon_chain_config.from_the_scratch,
    )
    fund_lagoon_vault(
        web3=web3_ethereum,
        vault_address=deployment.vault.address,
        asset_manager=deployer.address,
        test_account_with_balance=deployer.address,
        trading_strategy_module_address=deployment.trading_strategy_module.address,
        amount=SAFE_USDC,
        hot_wallet=deployer,
    )

    usdc_asset = AssetIdentifier(
        chain_id=1,
        address=USDC_NATIVE_TOKEN[1],
        token_symbol="USDC",
        decimals=6,
    )
    lighter_pair = create_lighter_exchange_account_pair(
        quote=usdc_asset,
        account_index=LIGHTER_ACCOUNT_INDEX,
    )
    state = State()
    sync_model = LagoonVaultSyncModel(
        vault=deployment.vault,
        hot_wallet=deployer,
        unit_testing=True,
    )
    sync_model.sync_initial(
        state,
        reserve_asset=usdc_asset,
        reserve_token_price=1.0,
    )
    open_exchange_account_position(
        state=state,
        strategy_cycle_at=native_datetime_utc_now(),
        pair=lighter_pair,
        reserve_currency=usdc_asset,
        reserve_amount=Decimal("0"),
    )

    # 2. Create and revalue a Lighter exchange-account position through the
    # production public-reader adapter. The distinct collateral/PnL values
    # prove the canonical total_asset_value is used rather than collateral.
    equity = type(
        "MockLighterEquity",
        (),
        {
            "collateral": LIGHTER_COLLATERAL,
            "unrealised_pnl": LIGHTER_UNREALISED_PNL,
            "get_total": lambda self: LIGHTER_TOTAL_EQUITY,
        },
    )()
    reader = mocker.patch(
        "tradeexecutor.exchange_account.lighter.fetch_lighter_total_equity",
        return_value=equity,
    )
    session = object()
    account_value_func = create_lighter_account_value_func(session)
    position = list(state.portfolio.open_positions.values())[0]
    cycle = native_datetime_utc_now()
    valuator = ExchangeAccountValuator(
        ExchangeAccountPricingModel(account_value_func),
        web3=web3_ethereum,
    )
    valuator(cycle, position)

    # 3. Reconcile the Safe and post NAV through the real Lagoon sync model.
    nav_func = create_lighter_vault_valuation_func(
        web3=web3_ethereum,
        safe_address=deployment.safe_address,
        reserve_asset=usdc_asset,
        account_index=LIGHTER_ACCOUNT_INDEX,
        session=session,
    )
    sync_model.calculate_valuation_func = nav_func
    nav_start_block = web3_ethereum.eth.block_number
    sync_model.sync_treasury(
        strategy_cycle_ts=cycle,
        state=state,
        post_valuation=True,
    )

    # 4. Verify the state components, on-chain reserve and posted total assets.
    reserve_position = state.portfolio.get_default_reserve_position()
    assert reserve_position.quantity == pytest.approx(SAFE_USDC)
    assert position.get_value() == pytest.approx(LIGHTER_TOTAL_EQUITY)
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(0)
    assert usdc.fetch_balance_of(deployment.safe_address) == pytest.approx(SAFE_USDC)

    # With an empty investor queue the sync model intentionally posts NAV but
    # does not call settleDeposit(); ``totalAssets`` remains the last settled
    # value. Read the emitted pending ``newTotalAssets`` event instead (the
    # deployed v0.5 ABI does not expose a getter for that storage slot).
    nav_logs = fetch_vault_settlement_logs(
        web3=web3_ethereum,
        address=deployment.vault.address,
        topic0_list=[Web3.to_hex(Web3.keccak(text="NewTotalAssetsUpdated(uint256)"))],
        start_block=nav_start_block,
        end_block=web3_ethereum.eth.block_number,
        use_hypersync=False,
    )
    assert nav_logs
    posted_raw = int.from_bytes(bytes(nav_logs[-1]["data"]), byteorder="big")
    posted_nav = usdc.convert_to_decimals(posted_raw)
    assert posted_nav == pytest.approx(SAFE_USDC + LIGHTER_TOTAL_EQUITY)
    assert reader.call_count >= 2
    for call in reader.call_args_list:
        assert call.args[1] == LIGHTER_ACCOUNT_INDEX

    # 5. Reject a negative public-equity response before another NAV transaction.
    # The response is mocked because Anvil cannot make Lighter's off-chain
    # sequencer report impossible account equity for this local vault.
    negative_equity = type(
        "MockNegativeLighterEquity",
        (),
        {"get_total": lambda self: Decimal("-0.01")},
    )()
    reader.return_value = negative_equity
    block_before_negative = web3_ethereum.eth.block_number
    with pytest.raises(NegativeLighterEquityError):
        sync_model.sync_treasury(
            strategy_cycle_ts=cycle,
            state=state,
            post_valuation=True,
        )
    assert web3_ethereum.eth.block_number == block_before_negative
