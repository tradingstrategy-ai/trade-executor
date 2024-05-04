"""Value model for 1delta trade based on Uniswap v3 market price.
"""
import datetime

from tradeexecutor.ethereum.one_delta.one_delta_live_pricing import OneDeltaLivePricing
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.ethereum.eth_valuation import EthereumPoolRevaluator
from tradeexecutor.state.valuation import ValuationUpdate


class OneDeltaPoolRevaluator(EthereumPoolRevaluator):
    """1delta position valuation."""

    def __init__(self, pricing_model: OneDeltaLivePricing):
        assert isinstance(pricing_model, OneDeltaLivePricing)
        super().__init__(pricing_model)

    def __call__(
        self,
        ts: datetime.datetime,
        position: TradingPosition,
    ) -> ValuationUpdate:

        pair = position.pair

        if pair.is_leverage():
            old_price = position.last_token_price
            old_value = position.get_value()

            quantity = abs(position.get_quantity())
            assert quantity > 0
            price_structure = self.pricing_model.get_sell_price(ts, pair.get_pricing_pair(), quantity)
            new_price = price_structure.price
            new_value = position.revalue_base_asset(ts, float(new_price))

            evt = ValuationUpdate(
                created_at=ts,
                position_id=position.position_id,
                valued_at=ts,
                new_price=new_price,
                new_value=new_value,
                old_price=old_price,
                old_value=old_value,
            )
        elif pair.is_credit_supply():
            # The position should be valued in `sync_interests`
            # so we just return the latest synced data here.

            loan = position.loan
            new_price = loan.collateral.last_usd_price
            valued_at = loan.collateral.last_pricing_at
            new_value = loan.get_net_asset_value()
            block_number = loan.collateral_interest.last_updated_block_number

            evt = ValuationUpdate(
                created_at=ts,
                position_id=position.position_id,
                valued_at=valued_at,
                new_price=new_price,
                new_value=new_value,
                block_number=block_number,
            )
        else:
            raise ValueError(f"Unknown position kind: {pair.kind}")

        position.valuation_updates.append(evt)

        return evt


def one_delta_valuation_factory(pricing_model):
    return OneDeltaPoolRevaluator(pricing_model)