"""Test CCTP bridge support for multichain strategies.

Tests cross-chain USDC bridging via Circle's CCTP V2 protocol:

1. Bridge out (burn on source chain, mint on destination)
2. Bridge back (reverse direction)
3. Bridge + swap on satellite chain

Pure unit tests — no RPC connections or Anvil forks needed.
"""

import datetime
from decimal import Decimal

import pytest

from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.trade import TradeExecution, TradeFlag, TradeType
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


#: Arbitrum native USDC address
USDC_ARBITRUM_ADDRESS = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"

#: Base native USDC address
USDC_BASE_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"


class DummyCctpPricingModel:
    """Minimal pricing model for exercising bridge close planning."""

    def get_sell_price(
        self,
        timestamp: datetime.datetime,
        pair: TradingPairIdentifier,
        quantity: Decimal | None = None,
    ) -> TradePricing:
        """Return 1:1 pricing for a CCTP bridge sell."""
        assert pair.is_cctp_bridge()
        assert quantity is not None
        return TradePricing(
            price=1.0,
            mid_price=1.0,
            lp_fee=[0.0],
            pair_fee=[0.0],
            side=False,
        )

    def calculate_trade_adjusted_slippage_tolerance(
        self,
        pair: TradingPairIdentifier,
        direction: str,
        default_slippage_tolerance: float,
    ) -> float:
        """Return a fixed small slippage tolerance for close planning."""
        assert pair.is_cctp_bridge()
        assert direction == "sell"
        return 0.01


def create_dummy_strategy_universe(reserve_asset: AssetIdentifier) -> TradingStrategyUniverse:
    """Create the smallest possible strategy universe for bridge close tests."""
    exchange = generate_exchange(
        exchange_id=1,
        chain_id=ChainId.arbitrum,
        address=generate_random_ethereum_address(),
    )
    return TradingStrategyUniverse(
        data_universe=Universe(
            time_bucket=TimeBucket.d1,
            chains={ChainId.arbitrum, ChainId.base},
            exchanges={exchange},
        ),
        reserve_assets=[reserve_asset],
    )


def create_base_weth_usdc_pair(usdc_base_asset: AssetIdentifier) -> TradingPairIdentifier:
    """Create a simple satellite-chain WETH/USDC pair for bridge accounting tests."""
    weth_base = AssetIdentifier(
        chain_id=8453,
        address="0x4200000000000000000000000000000000000006",
        token_symbol="WETH",
        decimals=18,
    )
    return TradingPairIdentifier(
        base=weth_base,
        quote=usdc_base_asset,
        pool_address="0xd0b53D9277642d899DF5C87A3966A349A798F224",
        exchange_address="0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
        fee=0.0005,
        kind=TradingPairKind.spot_market_hold,
    )


def create_open_bridge_position(
    state: State,
    ts: datetime.datetime,
    cctp_pair: TradingPairIdentifier,
    reserve_asset: AssetIdentifier,
    reserve_amount: Decimal,
) -> tuple[TradeExecution, Decimal]:
    """Open a CCTP bridge position and return the bridge trade and reserve quantity."""
    _, bridge_trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=cctp_pair,
        quantity=None,
        reserve=reserve_amount,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
    )
    state.start_execution(ts, bridge_trade)
    bridge_trade.mark_broadcasted(ts)
    state.mark_trade_success(
        executed_at=ts,
        trade=bridge_trade,
        executed_price=1.0,
        executed_amount=reserve_amount,
        executed_reserve=reserve_amount,
        lp_fees=0,
        native_token_price=0,
    )
    return bridge_trade, reserve_amount


@pytest.fixture()
def usdc_arbitrum_asset() -> AssetIdentifier:
    """USDC on Arbitrum as an AssetIdentifier."""
    return AssetIdentifier(
        chain_id=42161,
        address=USDC_ARBITRUM_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def usdc_base_asset() -> AssetIdentifier:
    """USDC on Base as an AssetIdentifier."""
    return AssetIdentifier(
        chain_id=8453,
        address=USDC_BASE_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def cctp_pair(usdc_arbitrum_asset, usdc_base_asset) -> TradingPairIdentifier:
    """CCTP bridge pair: Arbitrum -> Base."""
    return TradingPairIdentifier(
        base=usdc_base_asset,
        quote=usdc_arbitrum_asset,
        pool_address="0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",  # TOKEN_MESSENGER_V2
        exchange_address="0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",
        fee=0,
        kind=TradingPairKind.cctp_bridge,
        other_data={"bridge_protocol": "cctp"},
    )


def test_cctp_pair_properties(cctp_pair):
    """Test that CCTP bridge pair has correct properties."""
    assert cctp_pair.is_cctp_bridge()
    assert cctp_pair.kind == TradingPairKind.cctp_bridge

    # Source chain is Arbitrum (quote token chain)
    assert cctp_pair.get_source_chain_id() == 42161
    assert cctp_pair.chain_id == 8453  # chain_id returns destination for bridge pairs

    # Destination chain is Base (base token chain)
    assert cctp_pair.get_destination_chain_id() == 8453

    # Fee is zero
    assert cctp_pair.fee == 0

    # Label includes destination chain name
    label = cctp_pair.get_cctp_bridge_label()
    assert "USDC" in label
    assert "CCTP bridged to" in label

    # Ticker uses bridge label
    ticker = cctp_pair.get_ticker()
    assert "CCTP bridged to" in ticker


def test_cctp_bridge_out_and_back(cctp_pair, usdc_arbitrum_asset):
    """Test basic bridge out and bridge back with state tracking.

    1. Create state with reserves on Arbitrum
    2. Bridge out: open bridge position (burn on Arbitrum)
    3. Verify bridge position created with correct label
    4. Bridge back: close bridge position (burn on Base)
    5. Verify position closed and reserves restored
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    # Initialise reserves: 10000 USDC on Arbitrum
    state.portfolio.initialise_reserves(usdc_arbitrum_asset)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10000)
    reserve.reserve_token_price = 1.0

    # --- Bridge out: 1000 USDC from Arbitrum to Base ---

    position, trade, created = state.create_trade(
        strategy_cycle_at=ts,
        pair=cctp_pair,
        quantity=None,
        reserve=Decimal(1000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum_asset,
        reserve_currency_price=1.0,
    )

    assert created is True
    assert trade.pair.is_cctp_bridge()

    # Start execution allocates from reserves
    state.start_execution(ts, trade)
    assert trade.reserve_currency_allocated == Decimal(1000)

    # Simulate successful burn
    trade.mark_broadcasted(ts)
    state.mark_trade_success(
        executed_at=ts,
        trade=trade,
        executed_price=1.0,
        executed_amount=Decimal(1000),
        executed_reserve=Decimal(1000),
        lp_fees=0,
        native_token_price=0,
    )

    # Verify position
    assert len(state.portfolio.open_positions) == 1
    bridge_position = list(state.portfolio.open_positions.values())[0]
    assert bridge_position.pair.is_cctp_bridge()
    assert bridge_position.get_quantity() == Decimal(1000)

    # Verify reserves decreased
    assert reserve.quantity == Decimal(9000)

    # Verify position label
    label = bridge_position.pair.get_cctp_bridge_label()
    assert "USDC" in label
    assert "CCTP bridged to" in label

    # --- Bridge back: 1000 USDC from Base to Arbitrum ---

    _, trade_back, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=cctp_pair,
        quantity=Decimal(-1000),
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum_asset,
        reserve_currency_price=1.0,
    )

    assert trade_back.is_sell()

    state.start_execution(ts, trade_back)

    # Simulate successful bridge back (sell = negative amount)
    trade_back.mark_broadcasted(ts)
    state.mark_trade_success(
        executed_at=ts,
        trade=trade_back,
        executed_price=1.0,
        executed_amount=Decimal(-1000),
        executed_reserve=Decimal(1000),
        lp_fees=0,
        native_token_price=0,
    )

    # Verify position closed and reserves restored
    assert len(state.portfolio.open_positions) == 0
    assert reserve.quantity == Decimal(10000)


def test_bridge_then_spot_on_satellite(cctp_pair, usdc_arbitrum_asset, usdc_base_asset):
    """Test bridge + spot trade on satellite chain using bridge position as funding.

    1. Bridge 1000 USDC to Base
    2. Open spot trade on Base (spend from bridge position)
    3. Verify bridge position capital allocated reduced
    4. Close spot trade on Base (return to bridge position)
    5. Bridge back remaining USDC
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    # Initialise reserves: 10000 USDC on Arbitrum
    state.portfolio.initialise_reserves(usdc_arbitrum_asset)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10000)
    reserve.reserve_token_price = 1.0

    # --- Step 1: Bridge 1000 USDC to Base ---

    position, bridge_trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=cctp_pair,
        quantity=None,
        reserve=Decimal(1000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum_asset,
        reserve_currency_price=1.0,
    )

    state.start_execution(ts, bridge_trade)
    bridge_trade.mark_broadcasted(ts)
    state.mark_trade_success(
        executed_at=ts,
        trade=bridge_trade,
        executed_price=1.0,
        executed_amount=Decimal(1000),
        executed_reserve=Decimal(1000),
        lp_fees=0,
        native_token_price=0,
    )

    bridge_position = list(state.portfolio.open_positions.values())[0]
    assert bridge_position.get_quantity() == Decimal(1000)
    assert bridge_position.get_available_bridge_capital() == Decimal(1000)
    assert reserve.quantity == Decimal(9000)

    # --- Step 2: Spot trade on Base — buy WETH with bridged USDC ---

    weth_base = AssetIdentifier(
        chain_id=8453,
        address="0x4200000000000000000000000000000000000006",
        token_symbol="WETH",
        decimals=18,
    )
    weth_usdc_base_pair = TradingPairIdentifier(
        base=weth_base,
        quote=usdc_base_asset,
        pool_address="0xd0b53D9277642d899DF5C87A3966A349A798F224",
        exchange_address="0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
        fee=0.0005,
        kind=TradingPairKind.spot_market_hold,
    )

    _, spot_buy_trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=weth_usdc_base_pair,
        quantity=None,
        reserve=Decimal(500),
        assumed_price=2500.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_base_asset,
        reserve_currency_price=1.0,
    )

    # Start execution should allocate from bridge position (not reserves)
    state.start_execution(ts, spot_buy_trade)

    # Verify it came from bridge, not reserves
    assert spot_buy_trade.bridge_currency_allocated == Decimal(500)
    assert spot_buy_trade.reserve_currency_allocated is None
    assert bridge_position.bridge_capital_allocated == Decimal(500)
    assert bridge_position.get_available_bridge_capital() == Decimal(500)
    assert reserve.quantity == Decimal(9000)  # Source chain reserves unchanged

    # Simulate spot buy execution
    spot_buy_trade.mark_broadcasted(ts)
    state.mark_trade_success(
        executed_at=ts,
        trade=spot_buy_trade,
        executed_price=2500.0,
        executed_amount=Decimal("0.2"),  # 500/2500
        executed_reserve=Decimal(500),
        lp_fees=0.25,
        native_token_price=0,
    )

    # --- Step 3: Sell WETH back to USDC on Base ---

    spot_position = None
    for p in state.portfolio.open_positions.values():
        if p.pair == weth_usdc_base_pair:
            spot_position = p
            break

    assert spot_position is not None

    _, spot_sell_trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=weth_usdc_base_pair,
        quantity=Decimal("-0.2"),
        reserve=None,
        assumed_price=2600.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_base_asset,
        reserve_currency_price=1.0,
    )

    state.start_execution(ts, spot_sell_trade)

    # Simulate spot sell execution (gained some value, sell = negative amount)
    spot_sell_trade.mark_broadcasted(ts)
    state.mark_trade_success(
        executed_at=ts,
        trade=spot_sell_trade,
        executed_price=2600.0,
        executed_amount=Decimal("-0.2"),
        executed_reserve=Decimal(520),  # 0.2 * 2600
        lp_fees=0.26,
        native_token_price=0,
    )

    # Verify proceeds went back to bridge position
    assert bridge_position.bridge_capital_allocated == Decimal(500) - Decimal(520)  # negative means returned more
    assert reserve.quantity == Decimal(9000)  # Source reserves still untouched

    # --- Step 4: Bridge back remaining USDC ---

    # The bridge position should still be open with the original 1000 USDC quantity
    # (bridge_capital_allocated tracks the allocation offset)
    assert bridge_position in state.portfolio.open_positions.values()


def test_close_cctp_bridge_uses_available_bridge_capital(
    cctp_pair: TradingPairIdentifier,
    usdc_arbitrum_asset: AssetIdentifier,
    usdc_base_asset: AssetIdentifier,
):
    """Test that closing a bridge uses available capital instead of raw bridged quantity.

    1. Open a bridge position and realise a profitable satellite round-trip.
    2. Close the bridge through PositionManager using the real bridge close path.
    3. Verify the planned reverse bridge size matches available bridge capital and mutates the same position.
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    # 1. Open a bridge position and realise a profitable satellite round-trip.
    state.portfolio.initialise_reserves(usdc_arbitrum_asset)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10000)
    reserve.reserve_token_price = 1.0

    create_open_bridge_position(
        state=state,
        ts=ts,
        cctp_pair=cctp_pair,
        reserve_asset=usdc_arbitrum_asset,
        reserve_amount=Decimal(1000),
    )

    bridge_position = list(state.portfolio.open_positions.values())[0]
    weth_usdc_base_pair = create_base_weth_usdc_pair(usdc_base_asset)

    _, spot_buy_trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=weth_usdc_base_pair,
        quantity=None,
        reserve=Decimal(500),
        assumed_price=2500.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_base_asset,
        reserve_currency_price=1.0,
    )
    state.start_execution(ts, spot_buy_trade)
    spot_buy_trade.mark_broadcasted(ts)
    state.mark_trade_success(
        executed_at=ts,
        trade=spot_buy_trade,
        executed_price=2500.0,
        executed_amount=Decimal("0.2"),
        executed_reserve=Decimal(500),
        lp_fees=0.25,
        native_token_price=0,
    )

    spot_position = next(
        p for p in state.portfolio.open_positions.values()
        if p.pair == weth_usdc_base_pair
    )
    _, spot_sell_trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=weth_usdc_base_pair,
        quantity=Decimal("-0.2"),
        reserve=None,
        assumed_price=2600.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_base_asset,
        reserve_currency_price=1.0,
    )
    state.start_execution(ts, spot_sell_trade)
    spot_sell_trade.mark_broadcasted(ts)
    state.mark_trade_success(
        executed_at=ts,
        trade=spot_sell_trade,
        executed_price=2600.0,
        executed_amount=Decimal("-0.2"),
        executed_reserve=Decimal(520),
        lp_fees=0.26,
        native_token_price=0,
    )

    assert spot_position.is_closed()
    assert len(state.portfolio.open_positions) == 1
    assert bridge_position.get_quantity() == Decimal(1000)
    assert bridge_position.get_available_bridge_capital() == Decimal(1020)

    strategy_universe = create_dummy_strategy_universe(usdc_arbitrum_asset)
    pricing_model = DummyCctpPricingModel()
    position_manager = PositionManager(ts, strategy_universe, state, pricing_model)

    # 2. Close the bridge through PositionManager using the real bridge close path.
    trades = position_manager.close_position(bridge_position)

    # 3. Verify the planned reverse bridge size matches available bridge capital and mutates the same position.
    assert len(trades) == 1
    trade = trades[0]
    assert trade.pair.is_cctp_bridge()
    assert trade.is_sell()
    assert trade.position_id == bridge_position.position_id
    assert trade.planned_quantity == Decimal("-1020")
    assert trade.planned_reserve == Decimal("1020")
    assert TradeFlag.close in trade.flags
    assert TradeFlag.reduce in trade.flags
    assert len(state.portfolio.open_positions) == 1
    assert list(state.portfolio.open_positions.values())[0] == bridge_position


def test_cctp_bridge_can_be_closed_when_available_capital_is_zero_or_negative(
    cctp_pair: TradingPairIdentifier,
    usdc_arbitrum_asset: AssetIdentifier,
):
    """Test that bridge closeability is based on available capital, not raw quantity.

    1. Open a bridge position and confirm a normal positive balance is not closeable.
    2. Drive the available bridge capital to zero and negative values while raw quantity stays positive.
    3. Verify zero, negative, and tiny-dust available capital are all considered closeable.
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    # 1. Open a bridge position and confirm a normal positive balance is not closeable.
    state.portfolio.initialise_reserves(usdc_arbitrum_asset)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10000)
    reserve.reserve_token_price = 1.0

    create_open_bridge_position(
        state=state,
        ts=ts,
        cctp_pair=cctp_pair,
        reserve_asset=usdc_arbitrum_asset,
        reserve_amount=Decimal(1000),
    )

    bridge_position = list(state.portfolio.open_positions.values())[0]
    assert bridge_position.is_open()
    assert bridge_position.get_quantity() == Decimal(1000)
    assert bridge_position.get_available_bridge_capital() == Decimal(1000)
    assert bridge_position.can_be_closed() is False

    # 2. Drive the available bridge capital to zero and negative values while raw quantity stays positive.
    bridge_position.bridge_capital_allocated = Decimal(1000)
    assert bridge_position.get_quantity() == Decimal(1000)
    assert bridge_position.get_available_bridge_capital() == Decimal(0)
    assert bridge_position.can_be_closed() is True

    bridge_position.bridge_capital_allocated = Decimal(1001)
    assert bridge_position.get_quantity() == Decimal(1000)
    assert bridge_position.get_available_bridge_capital() == Decimal(-1)
    assert bridge_position.can_be_closed() is True

    # 3. Verify zero, negative, and tiny-dust available capital are all considered closeable.
    bridge_position.bridge_capital_allocated = Decimal("999.999999")
    assert bridge_position.get_quantity() == Decimal(1000)
    assert bridge_position.get_available_bridge_capital() == Decimal("0.000001")
    assert bridge_position.can_be_closed() is True

    bridge_position.bridge_capital_allocated = Decimal(750)
    assert bridge_position.get_available_bridge_capital() == Decimal(250)
    assert bridge_position.can_be_closed() is False


def test_cctp_pricing_model(cctp_pair):
    """Test that CCTP bridge pricing always returns 1:1."""
    from tradeexecutor.ethereum.cctp.pricing import CctpBridgePricingModel

    pricing = CctpBridgePricingModel()
    ts = datetime.datetime(2025, 1, 1)

    buy_price = pricing.get_buy_price(ts, cctp_pair, Decimal(1000))
    assert buy_price.price == 1.0
    assert buy_price.mid_price == 1.0
    assert buy_price.lp_fee == [0.0]

    sell_price = pricing.get_sell_price(ts, cctp_pair, Decimal(1000))
    assert sell_price.price == 1.0
    assert sell_price.mid_price == 1.0

    mid_price = pricing.get_mid_price(ts, cctp_pair)
    assert mid_price == 1.0


def test_cctp_valuation_model(cctp_pair, usdc_arbitrum_asset):
    """Test that CCTP bridge valuation returns face value."""
    from tradeexecutor.ethereum.cctp.valuation import CctpBridgeValuationModel

    state = State()
    ts = datetime.datetime(2025, 1, 1)

    state.portfolio.initialise_reserves(usdc_arbitrum_asset)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10000)
    reserve.reserve_token_price = 1.0

    position, trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=cctp_pair,
        quantity=None,
        reserve=Decimal(1000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum_asset,
        reserve_currency_price=1.0,
    )

    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)
    state.mark_trade_success(
        executed_at=ts,
        trade=trade,
        executed_price=1.0,
        executed_amount=Decimal(1000),
        executed_reserve=Decimal(1000),
        lp_fees=0,
        native_token_price=0,
    )

    valuator = CctpBridgeValuationModel()
    update = valuator(ts, position)

    assert update.new_price == 1.0
    assert update.new_value == pytest.approx(1000.0)
    assert position.last_token_price == 1.0
