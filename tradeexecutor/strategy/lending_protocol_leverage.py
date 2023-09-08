"""Lendindg protocol leveraged.

- Various helpers related to lending protocol leverage
"""

import datetime
from _decimal import Decimal
from dataclasses import dataclass
from typing import Tuple

from tradeexecutor.state.identifier import AssetWithTrackedValue, AssetType
from tradeexecutor.state.interest import Interest
from tradeexecutor.state.loan import Loan
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount, LeverageMultiplier, Percent, USDollarPrice





def create_credit_supply_loan(
    position: "tradeexecutor.state.position.TradingPosition",
    trade: TradeExecution,
    timestamp: datetime.datetime,
):
    """Create a loan that supplies credit to a lending protocol.

    This is a loan with

    - Collateral only

    - Borrowed is ``None``
    """

    assert trade.is_credit_supply()
    assert not position.loan

    pair = position.pair
    assert pair.is_credit_supply()

    # aToken

    #
    # The expected collateral
    # is our collateral allocation (reserve)
    # and whatever more collateral we get for selling the shorted token
    #

    collateral = AssetWithTrackedValue(
        asset=pair.base,  # aUSDC token is the base pair for credit supply positions
        last_usd_price=trade.reserve_currency_exchange_rate,
        last_pricing_at=datetime.datetime.utcnow(),
        quantity=trade.planned_reserve,
    )

    loan = Loan(
        pair=trade.pair,
        collateral=collateral,
        collateral_interest=Interest.open_new(trade.planned_reserve, timestamp),
        borrowed=None,
        borrowed_interest=None,
    )

    # Sanity check
    loan.check_health()

    return loan


def update_credit_supply_loan(
    position: "tradeexecutor.state.position.TradingPosition",
    trade: TradeExecution,
    timestamp: datetime.datetime,
):
    """Close/increase/reduce credit supply loan.

    """

    assert trade.is_credit_supply()

    pair = position.pair
    assert pair.is_credit_supply()

    loan = position.loan
    assert loan

    loan.collateral.change_quantity_and_value(
        trade.planned_quantity,
        trade.reserve_currency_exchange_rate,
        trade.opened_at,
        allow_negative=True,
    )

    # Sanity check
    loan.check_health()

    return loan


def create_short_loan(
    position: "tradeexecutor.state.position.TradingPosition",
    trade: TradeExecution,
    timestamp: datetime.datetime,
) -> Loan:
    """Create the loan data tracking for short position.

    - Check that the information looks correct for a short position.

    - Populates :py:class:`Loan` data structure.

    - We use assumed prices. The actual execution prices may differ
      and must be populated to `trade.executed_loan`.
    """

    assert trade.is_short()
    assert len(position.trades) == 1, "Can be only called when position is opening"

    assert not position.loan, f"loan already set"

    pair = trade.pair

    assert pair.base.underlying, "Base token lacks underlying asset"
    assert pair.quote.underlying, "Quote token lacks underlying asset"

    assert pair.base.type == AssetType.borrowed, f"Trading pair base asset is not borrowed: {pair.base}"
    assert pair.quote.type == AssetType.collateral, f"Trading pair quote asset is not collateral: {pair.quote}"

    assert pair.quote.underlying.is_stablecoin(), f"Only stablecoin collateral supported for shorts: {pair.quote}"

    # Extra checks when position is opened
    assert trade.planned_quantity < 0, f"Short position must open with a sell with negative quantity, got: {trade.planned_quantity}"

    if not trade.planned_collateral_allocation:
        assert trade.planned_reserve > 0, f"Collateral must be positive: {trade.planned_reserve}"

    # vToken
    borrowed = AssetWithTrackedValue(
        asset=pair.base,
        last_usd_price=trade.planned_price,
        last_pricing_at=datetime.datetime.utcnow(),
        quantity=abs(trade.planned_quantity),
        created_strategy_cycle_at=trade.strategy_cycle_at,
    )

    # aToken

    #
    # The expected collateral
    # is our collateral allocation (reserve)
    # and whatever more collateral we get for selling the shorted token
    #

    collateral = AssetWithTrackedValue(
        asset=pair.quote,
        last_usd_price=trade.reserve_currency_exchange_rate,
        last_pricing_at=datetime.datetime.utcnow(),
        quantity=trade.planned_reserve + trade.planned_collateral_allocation + trade.planned_collateral_consumption,
    )

    loan = Loan(
        pair=trade.pair,
        collateral=collateral,
        borrowed=borrowed,
        collateral_interest=Interest.open_new(collateral.quantity, timestamp),
        borrowed_interest=Interest.open_new(borrowed.quantity, timestamp),
    )

    # Sanity check
    loan.check_health()

    return loan


def plan_loan_update_for_short(
    loan: Loan,
    position: "tradeexecutor.state.position.TradingPosition",
    trade: TradeExecution,
):
    """Update the loan data tracking for short position.

    - Check that the information looks correct for a short position.

    """
    assert trade.is_short()
    assert len(position.trades) > 1, "Can be only called when closing/reducing/increasing/position"

    planned_collateral_consumption = trade.planned_collateral_consumption or Decimal(0)
    planned_collateral_allocation = trade.planned_collateral_allocation or Decimal(0)

    loan.collateral.change_quantity_and_value(
        planned_collateral_consumption + planned_collateral_allocation,
        trade.reserve_currency_exchange_rate,
        trade.opened_at,
    )

    # In short position, positive value reduces the borrowed amount
    loan.borrowed.change_quantity_and_value(
        -trade.planned_quantity,
        trade.planned_price,
        trade.opened_at,
        # Because of interest events, and the fact that we need
        # to pay the interest back on closing the loan,
        # the tracked underlying amount can go negative when closing a short
        # position
        allow_negative=True,
    )

    # Sanity check
    if loan.borrowed.quantity > 0:
        loan.check_health()

    return loan


def calculate_sizes_for_leverage(
    starting_reserve: USDollarAmount,
    leverage: LeverageMultiplier,
) -> Tuple[USDollarAmount, USDollarAmount]:
    """Calculate the collateral and borrow loan size to hit the target leverage with a starting capital.

    - When calculating the loan size using this function,
      the loan net asset value will be the same as starting capital

    - Because loan net asset value is same is deposited reserve,
      portfolio total NAV stays intact

    Notes:

    .. code-block:: text

            col / (col - borrow) = leverage
            col = (col - borrow) * leverage
            col = col * leverage - borrow * leverage
            col - col * leverage = - borrow * levereage
            col(1 - leverage) = - borrow * leverage
            col = -(borrow * leverage) / (1 - leverage)

            # Calculate leverage for 4x and 1000 USD collateral
            col - borrow = 1000
            col = 1000
            leverage = 3

            col / (col - borrow) = 3
            3(col - borrow) = col
            3borrow = 3col - col
            borrow = col - col/3

            col / (col - (col - borrow)) = leverage
            col / borrow = leverage
            borrow = leverage * 1000


    :param starting_reserve:
        Initial deposit in lending protocol

    :return:
        Tuple (borrow value, collateral value) in dollars
    """

    collateral_size = starting_reserve * leverage
    borrow_size = (collateral_size - (collateral_size / leverage))
    return borrow_size, collateral_size


@dataclass
class LeverageEstimate:
    """Estimate token quantities and fees for a leverage position.

    A helper class to make sense out of fees when doing leveraged trading on 1delta / Aave.

    - When increasing short, the fees are allocated to the collateral we need to borrow

    **Opening short**

    - Doing 3x ETH short
        - Start with 10 USDC
        - Deposit in Aave
        - ETH price is 1,500 USD/ETH
        - Using swap exact out method
        - The short should be 20k USD worth of ETH, 30k USD collateral

    - Deposit 10 USDC to Aave

    - Open position with 1delta protocol contract
        - 1delta takes inputs
        - 1delta initiates swap for 0.0133333333333 WETH to 20 USDC (minus fees)
        - Uniswap v3 calls back 1delta
        - 1delta mints out USDC aToken from USDC we received from the swap
        - We have now total 10 + 19.99 USDC in Aave
        - 1delta borrows WETH for 0.0133333333333 WETH
        - Uniswap is happy has we have WETH we did not have at the start of the process

    - Fee calculations
        - Token in: Borrowed ETH (no fees) 13.3333333333
        - Token out: 19.99 USDC (0.01 USD paid in fees)
    - Final outcome
        - vWETH 0.0133333333
        - aUSDC 19.99

    Example transaction

    - https://dashboard.tenderly.co/tx/polygon/0xaf9bddedc174dc051abcdb28e6be6bf7f337ce73a9d9ba47bf51b42c04fe0df1?trace=0.0.0
    """

    #: Amount of USDC reserve we use for this position
    starting_reserve: Decimal

    #: What was the leverage multiplier we used
    leverage: LeverageMultiplier

    #: Amount of the borrowed token we short
    borrowed_quantity: Decimal

    #: What is the borrowed asset value in USD
    #:
    borrowed_value: USDollarAmount

    #: How much additional collateral we are going to take.
    #
    #: This is the output when we sell borrowed asset.
    #:
    additional_collateral_quantity: Decimal

    #: Amount of total collateral we have
    #:
    #: This is starting reserve +
    #:
    total_collateral_quantity: Decimal

    #: What's the price for the borrowed token.
    #:
    #: - Assume collateral = USDC
    #  - Assume USDC 1:1 with USD
    #:
    borrowed_asset_price: USDollarPrice

    #: What was our swap fee tier
    #:
    #: E.g ``0.0005`` for 5 BPS.
    fee_tier: Percent

    #: How much fees we are going to be.
    #:
    #: Estimate swap fees.
    #:
    lp_fees: USDollarAmount

    def __repr__(self):
        return f"<Leverage estimate\n" \
               f"    leverage: {self.leverage}\n" \
               f"    reserve allocated (USDC): {self.starting_reserve}\n" \
               f"    borrowed (vToken): {self.borrowed_quantity}\n" \
               f"    total collateral (aToken): {self.total_collateral_quantity}\n" \
               f"    LP fees: {self.lp_fees} USD\n" \
               f">\n"

    @staticmethod
    def open_short(
        starting_reserve: USDollarAmount | Decimal,
        leverage: LeverageMultiplier,
        borrowed_asset_price: USDollarAmount,
        fee: Percent = 0,
    ) -> "LeverageEstimate":
        """Get borrow and colleteral size for a loan in leverage protocol trading.

        See :py:func:`calculate_sizes_for_leverage`.

        .. note ::

            Short only. Stablecoin collateral only.

        Example:

        .. code-block:: python

            from tradeexecutor.strategy.lending_protocol_leverage import calculate_quantities_for_leverage

            # Start with 10 USD
            starting_capital = 10.0
            leverage = 3.0
            eth_price = 1629.43

            # This will borrow additional
            # - 20 USDC as collateral
            # - 0.01228645 WETH
            borrow_quantity, collateral_quantity = calculate_quantities_for_leverage(
                starting_capital,
                leverage,
                eth_price,
            )

            print(f"Borrowing {borrow_quantity:,.8f} token using {collateral_quantity:,.8f} USD as collateral")

        :param starting_capital:
            How much USDC we are going to deposit

        :param leverage:
            How much leverage we take

        :param token_price:
            What is the price of a token we short

        :param fee:
            What is the trading fee for swapping the borrowed asset to collateral

        :return:
            borrow quantity, collateral quantity for the constructed loan

        """

        assert isinstance(starting_reserve, Decimal)

        # Assume collateral is USDC
        total_collateral_quantity = starting_reserve * Decimal(leverage)
        borrow_value_usdc = (total_collateral_quantity - (total_collateral_quantity / Decimal(leverage)))

        additional_collateral_quantity_no_fee = total_collateral_quantity - starting_reserve
        swapped_out = additional_collateral_quantity_no_fee * (Decimal(1) - Decimal(fee))
        paid_fee = float(additional_collateral_quantity_no_fee) * fee

        borrow_quantity = borrow_value_usdc / Decimal(borrowed_asset_price)

        return LeverageEstimate(
            starting_reserve=Decimal(starting_reserve),
            leverage=leverage,
            borrowed_quantity=borrow_quantity,
            borrowed_value=float(borrow_value_usdc),
            additional_collateral_quantity=swapped_out,
            total_collateral_quantity=swapped_out + starting_reserve,
            borrowed_asset_price=borrowed_asset_price,
            fee_tier=fee,
            lp_fees=paid_fee,
        )
