"""Interest accruing functions."""
import datetime
from decimal import Decimal

from web3 import Web3
from eth_defi.abi import get_deployed_contract

from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdatePositionType, BalanceUpdateCause
from tradeexecutor.state.identifier import TradingPairKind, TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State


def get_onchain_atoken_amount(
    web3,
    lending_reserve_identifier: TradingPairIdentifier,
    wallet_address: str,
    block_number: int | None = None,
) -> Decimal:
    assert lending_reserve_identifier.kind == TradingPairKind.credit_supply

    # aToken is the base asset of this pair
    atoken_address = lending_reserve_identifier.base.address
    atoken_decimals = lending_reserve_identifier.base.decimals

    atoken_contract = get_deployed_contract(web3, "ERC20MockDecimals.json", Web3.to_checksum_address(atoken_address))

    raw_amount = atoken_contract.functions.balanceOf(wallet_address).call(block_identifier=block_number)

    return Decimal(raw_amount) / Decimal(10 ** atoken_decimals)


def update_credit_supply_interest(
    state: State,
    position: TradingPosition,
    new_atoken_amount: Decimal,
    event_at: datetime.datetime,
    block_number: int | None = None,
    tx_hash: int | None = None,
    log_index: int | None = None,
):
    """Poke credit supply position to increase its interest amount.

    :param event_at:
        Block mined timestamp

    """

    assert position.pair.kind == TradingPairKind.credit_supply
    assert position.is_open() and not position.is_frozen(), f"Cannot update interest for position {position}"

    portfolio = state.portfolio

    event_id = portfolio.allocate_balance_update_id()

    # We use USDC (not AUSDC) as the asset
    # for credit supply interest events
    asset = position.pair.quote

    assert asset.is_stablecoin(), f"Credit supply is currently supported for stablecoin assets with 1:1 USD price assumption. Got: {asset}"

    old_balance = position.interest.last_atoken_amount
    gained_interest = new_atoken_amount - old_balance
    usd_value = float(new_atoken_amount)

    assert 0 < gained_interest < 999, f"Unlikely gained_interest: {gained_interest}, old quantity: {position.quantity}, new quantity: {new_atoken_amount}"

    evt = BalanceUpdate(
        balance_update_id=event_id,
        position_type=BalanceUpdatePositionType.open_position,
        cause=BalanceUpdateCause.interest,
        asset=asset,
        block_mined_at=event_at,
        strategy_cycle_included_at=None,
        chain_id=asset.chain_id,
        old_balance=old_balance,
        usd_value=usd_value,
        quantity=gained_interest,
        owner_address=None,
        tx_hash=tx_hash,
        log_index=log_index,
        position_id=position.position_id,
        block_number=block_number,
    )

    position.add_balance_update_event(evt)

    # Update interest stats
    position.interest.last_accrued_interest = position.calculate_accrued_interest_tokens()
    position.interest.last_updated_at = datetime.datetime.utcnow()
    position.interest.last_event_at = event_at
    position.interest.last_updated_block_number = block_number
    position.interest.last_atoken_amount = new_atoken_amount
