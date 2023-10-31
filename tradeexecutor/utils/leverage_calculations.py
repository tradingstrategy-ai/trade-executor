"""Helpers for leverage calculations.

- Collateral/borrow size

- Fees needed to pay

- Liquidation price

"""
from decimal import Decimal
from dataclasses import dataclass

from tradeexecutor.state.identifier import TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.types import LeverageMultiplier, USDollarAmount, USDollarPrice, Percent


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
        - The short should be 20 USD worth of ETH, 29.99 USD collateral

    - Deposit 10 USDC to Aave

    - Open position with 1delta protocol contract
        - 1delta takes inputs
        - 1delta initiates swap for 0.0133333333333 WETH to 20 USDC (minus fees)
        - Uniswap v3 calls back 1delta
        - 1delta mints out USDC aToken from USDC we received from the swap
        - We have now total 10 (originak deposit) + 19.99 USDC (new loan) in Aave
        - 1delta borrows WETH for 0.0133333333333 WETH
        - Uniswap is happy has we have WETH we did not have at the start of the process

    - Fee calculations
        - Token in: Borrowed ETH (no fees) 13.3333333333
        - Token out: 19.99 USDC (0.01 USD paid in fees)
    - Final outcome
        - vWETH 0.0133333333
        - aUSDC 29.99

    - `Example transaction <https://dashboard.tenderly.co/tx/polygon/0xaf9bddedc174dc051abcdb28e6be6bf7f337ce73a9d9ba47bf51b42c04fe0df1?trace=0.0.0>`__

    ** Close short **

    - Closing 3x short as described above

    - Closing position with 1delta
        - Assume price is
        - Get exact wWETH debt amount: 0.0112236255452078143 vWETH
        - Start a swap process on Uniswap with WETH -> USDC for this amount
            - There is no WETH yet in this point
            - Uniswap will tell us how much USDC we will need later down the chain
        - Uniswap calls 1delta fallback called with incoming WETH from the swap
            - Aave loan is repaid with WETH
            - Uniswap tells os the USDC needed to cover the swap cost
            - Atoken USDC colleteral is converted back to USDC to cover the cost of the swap
            - The amount of USDC here is fee inclusive to match 0.0112236255452078143 vWETH,
              so it is wWETH price + fees

        - Total swap cost is = (0.0112236255452078143 / 0.9995) * ETH price

        - Fees are 0.0005 * (0.0112236255452078143 / 0.9995) * ETH price = ETH amount * (fee / (1-fee))

    - `Example transaction <https://dashboard.tenderly.co/tx/polygon/0x887bafca8fbe39a5188e385e638fa522146065b57e4ca6aa495926a840566272>`__
    """

    #: Amount of USDC reserve we use for this position.
    #:
    #: Set to 0 when closing/reducing short position as the position is covered from the collateral.
    #:
    starting_reserve: Decimal

    #: What was the leverage multiplier we used.
    #:
    #: Short open: This is the leverage the user desired.
    #:
    #: Short close/reduce: This is the leverage remaining.
    #:
    #:
    leverage: LeverageMultiplier

    #: Amount of the borrowed token we short.
    #:
    #: Positive if we increase our borrow.
    #:
    #: Negative if we reduce or borrow.
    #:
    borrowed_quantity: Decimal

    #: What is the borrowed asset value in USD
    #:
    borrowed_value: USDollarAmount

    #: How much additional collateral we are going to take.
    #
    #: This is the output when we buy/sell borrowed asset.
    #:
    #: Positive: We are selling WETH and adding this USDC to our debt.
    #:
    #: Negative: We are buying WETH and need to convert this much of collateral to USDC to match the
    #: cost.
    #:
    additional_collateral_quantity: Decimal

    #: Amount of total collateral we have
    #:
    #: Short open: This is starting reserve + additional collateral borrowed.
    #:
    #: Short close: This is remaining collateral after converting it to
    #: cover the trade to close the short.
    #:
    total_collateral_quantity: Decimal

    #: What is the total borrow amount after this ooperation
    #:
    total_borrowed_quantity: Decimal

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

    #: What is the liquidation price for this position.
    #: If the price goes below this, the loan is liquidated.
    #:
    liquidation_price: USDollarAmount | None = None

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
        shorting_pair: TradingPairIdentifier,
        fee: Percent = 0,
    ) -> "LeverageEstimate":
        """Get borrow and colleteral size for a loan in leverage protocol trading.

        See :py:func:`calculate_sizes_for_leverage`.

        .. note ::

            Short only. Stablecoin collateral only.

        Example:

        .. code-block:: python

            from tradeexecutor.strategy.lending_protocol_leverage import LeverageEstimate

            # Start with 10 USD
            starting_capital = 10.0
            leverage = 3.0
            eth_price = 1634.4869

            # This will borrow additional
            # - 20 USDC as collateral
            # - 0.01228645 WETH
            estimate = LeverageEstimate.open_short(
                starting_capital,
                leverage,
                eth_price,
                fee=0.0005,
            )

            print("Estimated amounts for the short:", estimate)

        Example output:

        .. code-block:: text

            Estimated amounts for the short: <Leverage estimate
                leverage: 3.0
                reserve allocated (USDC): 10
                borrowed (vToken): 0.01223625591615325807353339610
                total collateral (aToken): 29.98999999999999999979183318
                LP fees: 0.01 USD
            >

        :param starting_capital:
            How much USDC we are going to deposit

        :param leverage:
            How much leverage we take

        :param token_price:
            What is the price of a token we short

        :param shorting_pair:
            The synthetic trading pair for the lending pool short.

            With aToken and vToken.

        :param fee:
            What is the trading fee for swapping the borrowed asset to collateral.

            TODO: Use the fee from the trading pair.

        :return:
            borrow quantity, collateral quantity for the constructed loan

        """

        assert shorting_pair.kind == TradingPairKind.lending_protocol_short

        max_leverage = shorting_pair.get_max_leverage_at_open(side="short")
        assert leverage < max_leverage, f"Max short leverage for {shorting_pair.quote.underlying.token_symbol} is {max_leverage}, got {leverage}"

        if type(starting_reserve) == float:
            starting_reserve = Decimal(starting_reserve)

        # Assume collateral is USDC
        total_collateral_quantity = starting_reserve * (Decimal(leverage) + 1)
        borrow_value_usdc = total_collateral_quantity - starting_reserve

        additional_collateral_quantity_no_fee = total_collateral_quantity - starting_reserve
        swapped_out = additional_collateral_quantity_no_fee * (Decimal(1) - Decimal(fee))
        paid_fee = float(additional_collateral_quantity_no_fee) * fee

        borrow_quantity = borrow_value_usdc / Decimal(borrowed_asset_price)

        liquidation_price = calculate_liquidation_price(
            collateral_size=total_collateral_quantity,
            borrow_quantity=borrow_quantity,
            shorting_pair=shorting_pair,
        )

        return LeverageEstimate(
            starting_reserve=Decimal(starting_reserve),
            leverage=leverage,
            borrowed_quantity=borrow_quantity,
            borrowed_value=float(borrow_value_usdc),
            additional_collateral_quantity=swapped_out,
            total_collateral_quantity=swapped_out + starting_reserve,
            total_borrowed_quantity=borrow_quantity,
            borrowed_asset_price=borrowed_asset_price,
            fee_tier=fee,
            lp_fees=paid_fee,
            liquidation_price=liquidation_price,
        )

    @staticmethod
    def close_short(
        start_collateral: Decimal,
        start_borrowed: Decimal,
        close_size: Decimal,
        borrowed_asset_price: USDollarAmount,
        fee: Percent | None = 0,
    ) -> "LeverageEstimate":
        """Reduce or close short position.

        Calculate the trade mounts needed to close a short position.

        - Buy back shorted tokens

        - Release any collateral

        See :py:class:`LeverageEstimate` for fee calculation example.

        Example:

        .. code-block:: python

            estimate = LeverageEstimate.close_short(
                start_collateral=short_position.loan.collateral.quantity,
                start_borrowed=short_position.loan.borrowed.quantity,
                close_size=short_position.loan.borrowed.quantity,
                fee=weth_short_identifier_5bps.fee,
                borrowed_asset_price=1500.0,
            )

            assert estimate.leverage == 1.0  # Reduced USDC leverage to 1.0
            assert estimate.additional_collateral_quantity == pytest.approx(Decimal(-20010.00500250125062552103147))  # USDC needed to reduce from collateral to close position + fees
            assert estimate.borrowed_quantity == pytest.approx(Decimal(-13.33333333333333333333333333))  # How much ETH is bought to close the short
            assert estimate.total_collateral_quantity == pytest.approx(Decimal(9979.99499749874937427080171))  # Collateral left after closing the position
            assert estimate.total_borrowed_quantity == 0  # open vWETH debt left after close
            assert estimate.lp_fees == pytest.approx(10.005002501250626)

        We assume collateral is 1:1 USD.

        :param start_collateral:
            How much collateral we have at start.

        :param close_size:
            How much debt to reduce.

            Expressed in the amount of borrowed token quantity.

        """

        assert close_size > 0

        matching_usdc_amount = Decimal(borrowed_asset_price) * close_size

        if fee:
            assert fee > 0
            fee_decimal = Decimal(fee)
        else:
            fee_decimal = Decimal(0)

        matching_usdc_amount_with_fees = matching_usdc_amount / (Decimal(1) - fee_decimal)

        paid_fee = Decimal(matching_usdc_amount_with_fees * fee_decimal)

        total_collateral_quantity = start_collateral - matching_usdc_amount_with_fees
        total_borrowed_quantity = start_borrowed - close_size

        total_borrowed_usd = total_borrowed_quantity * Decimal(borrowed_asset_price)

        leverage = total_collateral_quantity / (total_collateral_quantity - total_borrowed_usd)

        return LeverageEstimate(
            starting_reserve=Decimal(0),
            leverage=float(leverage),
            borrowed_quantity=-close_size,
            borrowed_value=float(matching_usdc_amount),
            additional_collateral_quantity=-matching_usdc_amount_with_fees,
            total_collateral_quantity=total_collateral_quantity,
            total_borrowed_quantity=total_borrowed_quantity,
            borrowed_asset_price=borrowed_asset_price,
            fee_tier=fee,
            lp_fees=float(paid_fee),
        )
    

def calculate_sizes_for_leverage(
    starting_reserve: USDollarAmount,
    leverage: LeverageMultiplier,
) -> tuple[USDollarAmount, USDollarAmount, Decimal]:
    """Calculate the collateral and borrow loan size to hit the target leverage with a starting capital.

    - When calculating the loan size using this function,
      the loan net asset value will be the same as starting capital

    - Because loan net asset value is same is deposited reserve,
      portfolio total NAV stays intact

    Notes:

    .. code-block:: text

            nav = col - borrow
            leverage = borrow / nav
            leverage = col / nav - 1

            borrow = nav * leverage
            col = nav * leverage + nav
            col = nav * (leverage + 1)

            # Calculate leverage for 4x and 1000 USD nav (starting reserve)
            nav = 1000
            borrow = 1000 * 4 = 4000
            col = 1000 * 4 + 1000 = 5000
            col = 1000 * (4 + 1) = 5000
            col = 1000 + 4000 = 5000

    :param starting_reserve:
        Initial deposit in lending protocol

    :param shorting_pair:
        Leverage short trading pair

    :return:
        Tuple (borrow value, collateral value) in dollars
    """
    collateral_size = starting_reserve + (leverage + 1)
    borrow_size = collateral_size - starting_reserve

    return borrow_size, collateral_size



def calculate_liquidation_price(
    collateral_size: USDollarAmount,
    borrow_quantity: Decimal,
    shorting_pair: TradingPairIdentifier,
) -> USDollarAmount:
    """Calculate the liquidation price for a short position.

    lP = buy_USD * cfBuy / sell
        where:
        buy_USD: buy/deposit asset amount in USD
        sell: sell/borrow asset amount in sell currency

    - `See 1delta documentation <https://docs.1delta.io/lenders/metrics>`__.

    :param collateral_size:
        Collateral size in USD

    :param borrow_quantity:
        Borrow quantity in sell currency

    :param shorting_pair:
        Leverage short trading pair

    :return:
        Liquidation price in USD
    """
    assert shorting_pair.is_leverage()

    return Decimal(collateral_size) * Decimal(shorting_pair.get_collateral_factor()) / Decimal(borrow_quantity)
