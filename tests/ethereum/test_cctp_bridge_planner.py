"""Test CCTP bridge trade injection planner.

Verifies that :py:func:`inject_cctp_bridge_trades` correctly analyses
alpha model trades and injects the necessary bridge transfers:

- Bridge-out buys for satellite chain purchases
- Bridge-back sells for satellite chain sell proceeds
- Net flow calculation to avoid unnecessary round-trips
- No bridge injection for primary chain trades
- Correct handling of multiple satellite chains
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from tradeexecutor.ethereum.cctp.planner import inject_cctp_bridge_trades
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType


from tradingstrategy.chain import ChainId

USDC_ARBITRUM_ADDRESS = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
USDC_BASE_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
USDC_AVALANCHE_ADDRESS = "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E"

PRIMARY_CHAIN_ID = 42161  # Arbitrum
BASE_CHAIN_ID = 8453
AVALANCHE_CHAIN_ID = 43114
HYPERCORE_CHAIN_ID = ChainId.hypercore.value  # 9999

TS = datetime.datetime(2025, 6, 1)


@pytest.fixture()
def usdc_arbitrum() -> AssetIdentifier:
    """USDC on Arbitrum (primary chain reserve)."""
    return AssetIdentifier(
        chain_id=PRIMARY_CHAIN_ID,
        address=USDC_ARBITRUM_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def usdc_base() -> AssetIdentifier:
    """USDC on Base."""
    return AssetIdentifier(
        chain_id=BASE_CHAIN_ID,
        address=USDC_BASE_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def usdc_avalanche() -> AssetIdentifier:
    """USDC on Avalanche."""
    return AssetIdentifier(
        chain_id=AVALANCHE_CHAIN_ID,
        address=USDC_AVALANCHE_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def cctp_pair_base(
    usdc_arbitrum: AssetIdentifier,
    usdc_base: AssetIdentifier,
) -> TradingPairIdentifier:
    """CCTP bridge pair: Arbitrum -> Base."""
    return TradingPairIdentifier(
        base=usdc_base,
        quote=usdc_arbitrum,
        pool_address="0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",
        exchange_address="0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",
        fee=0,
        kind=TradingPairKind.cctp_bridge,
        other_data={"bridge_protocol": "cctp"},
    )


@pytest.fixture()
def cctp_pair_avalanche(
    usdc_arbitrum: AssetIdentifier,
    usdc_avalanche: AssetIdentifier,
) -> TradingPairIdentifier:
    """CCTP bridge pair: Arbitrum -> Avalanche."""
    return TradingPairIdentifier(
        base=usdc_avalanche,
        quote=usdc_arbitrum,
        pool_address="0x0000000000000000000000000000000000000077",
        exchange_address="0x0000000000000000000000000000000000000077",
        fee=0,
        kind=TradingPairKind.cctp_bridge,
        other_data={"bridge_protocol": "cctp"},
    )


@pytest.fixture()
def vault_pair_base(usdc_base: AssetIdentifier) -> TradingPairIdentifier:
    """A vault pair on Base (satellite chain)."""
    vault_token = AssetIdentifier(
        chain_id=BASE_CHAIN_ID,
        address="0x0000000000000000000000000000000000000011",
        token_symbol="vaultBASE",
        decimals=18,
    )
    return TradingPairIdentifier(
        base=vault_token,
        quote=usdc_base,
        pool_address="0x0000000000000000000000000000000000000022",
        exchange_address="0x0000000000000000000000000000000000000022",
        fee=0,
        kind=TradingPairKind.vault,
    )


@pytest.fixture()
def vault_pair_base_2(usdc_base: AssetIdentifier) -> TradingPairIdentifier:
    """A second vault pair on Base (satellite chain)."""
    vault_token = AssetIdentifier(
        chain_id=BASE_CHAIN_ID,
        address="0x0000000000000000000000000000000000000033",
        token_symbol="vaultBASE2",
        decimals=18,
    )
    return TradingPairIdentifier(
        base=vault_token,
        quote=usdc_base,
        pool_address="0x0000000000000000000000000000000000000044",
        exchange_address="0x0000000000000000000000000000000000000044",
        fee=0,
        kind=TradingPairKind.vault,
    )


@pytest.fixture()
def vault_pair_primary(usdc_arbitrum: AssetIdentifier) -> TradingPairIdentifier:
    """A vault pair on the primary chain (Arbitrum)."""
    vault_token = AssetIdentifier(
        chain_id=PRIMARY_CHAIN_ID,
        address="0x0000000000000000000000000000000000000055",
        token_symbol="vaultARB",
        decimals=18,
    )
    return TradingPairIdentifier(
        base=vault_token,
        quote=usdc_arbitrum,
        pool_address="0x0000000000000000000000000000000000000066",
        exchange_address="0x0000000000000000000000000000000000000066",
        fee=0,
        kind=TradingPairKind.vault,
    )


@pytest.fixture()
def vault_pair_avalanche(usdc_avalanche: AssetIdentifier) -> TradingPairIdentifier:
    """A vault pair on Avalanche (another satellite chain)."""
    vault_token = AssetIdentifier(
        chain_id=AVALANCHE_CHAIN_ID,
        address="0x0000000000000000000000000000000000000088",
        token_symbol="vaultAVAX",
        decimals=18,
    )
    return TradingPairIdentifier(
        base=vault_token,
        quote=usdc_avalanche,
        pool_address="0x0000000000000000000000000000000000000099",
        exchange_address="0x0000000000000000000000000000000000000099",
        fee=0,
        kind=TradingPairKind.vault,
    )


def _make_mock_universe(pairs: list[TradingPairIdentifier]):
    """Build a mock strategy universe that yields the given pairs."""
    mock = MagicMock()
    mock.iterate_pairs.return_value = pairs
    return mock


def _make_state_with_reserves(
    usdc_arbitrum: AssetIdentifier,
    reserve_amount: Decimal = Decimal(50_000),
) -> State:
    """Create a fresh state with reserves on the primary chain."""
    state = State()
    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = reserve_amount
    reserve.reserve_token_price = 1.0
    return state


def _execute_trade(
    state: State,
    trade: TradeExecution,
    ts: datetime.datetime,
    quantity: Decimal,
    reserve: Decimal,
    price: float = 1.0,
):
    """Simulate trade execution: start -> broadcast -> success."""
    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)
    trade.mark_success(
        executed_at=ts,
        executed_price=price,
        executed_quantity=quantity,
        executed_reserve=reserve,
        lp_fees=0.0,
        native_token_price=0.0,
    )


def test_inject_bridge_out_for_satellite_buy(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair_base: TradingPairIdentifier,
    vault_pair_base: TradingPairIdentifier,
):
    """Verify that a satellite chain buy injects a bridge-out (primary -> satellite).

    1. Create a state with reserves on the primary chain
    2. Create a vault buy trade on Base (satellite)
    3. Call inject_cctp_bridge_trades with a universe containing the bridge pair
    4. Assert exactly one bridge trade was injected
    5. Assert the bridge trade is a buy (bridge-out) for the correct amount
    6. Assert the bridge pair targets Base
    """
    # 1. Create state with reserves
    state = _make_state_with_reserves(usdc_arbitrum)
    universe = _make_mock_universe([cctp_pair_base, vault_pair_base])

    # 2. Create a vault buy on Base for 3000 USDC
    _, vault_buy, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=vault_pair_base,
        quantity=None,
        reserve=Decimal(3000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )

    # 3. Inject bridge trades
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[vault_buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    # 4. Assert one bridge trade was injected (original + bridge)
    assert len(result) == 2
    bridge_trade = result[1]

    # 5. Assert the bridge trade is a buy (bridge-out)
    assert bridge_trade.is_buy()
    assert bridge_trade.pair.is_cctp_bridge()
    assert bridge_trade.planned_reserve == Decimal("3000")

    # 6. Assert the bridge pair targets Base
    assert bridge_trade.pair.get_destination_chain_id() == BASE_CHAIN_ID


def test_inject_bridge_back_for_satellite_sell(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair_base: TradingPairIdentifier,
    vault_pair_base: TradingPairIdentifier,
):
    """Verify that a satellite chain sell injects a bridge-back (satellite -> primary).

    1. Create a state with reserves and an existing bridge position on Base
    2. Create a vault sell trade on Base
    3. Call inject_cctp_bridge_trades
    4. Assert exactly one bridge trade was injected
    5. Assert the bridge trade is a sell (bridge-back) for the correct amount
    6. Assert the bridge position is used for the sell
    """
    # 1. Create state with reserves and open a bridge position on Base
    state = _make_state_with_reserves(usdc_arbitrum)

    # First create a bridge-out to establish the bridge position
    _, bridge_out, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=cctp_pair_base,
        quantity=None,
        reserve=Decimal(10_000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )
    # Simulate the bridge-out completing
    _execute_trade(state, bridge_out, TS, Decimal(10_000), Decimal(10_000))

    # Verify bridge position exists
    bridge_pos = state.portfolio.get_bridge_position_for_chain(BASE_CHAIN_ID)
    assert bridge_pos is not None

    universe = _make_mock_universe([cctp_pair_base, vault_pair_base])

    # 2. Create a vault sell on Base for 5000 USDC
    # Open the vault position first
    _, vault_buy, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=vault_pair_base,
        quantity=None,
        reserve=Decimal(5000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )
    _execute_trade(state, vault_buy, TS, Decimal(5000), Decimal(5000))

    # Now create the sell trade
    vault_position = list(state.portfolio.open_positions.values())
    vault_pos = [p for p in vault_position if p.pair == vault_pair_base][0]
    _, vault_sell, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=vault_pair_base,
        quantity=Decimal(-5000),
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
        position=vault_pos,
        closing=True,
    )

    # 3. Inject bridge trades
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[vault_sell],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    # 4. Assert one bridge trade was injected
    assert len(result) == 2
    bridge_trade = result[1]

    # 5. Assert the bridge trade is a sell (bridge-back)
    assert bridge_trade.is_sell()
    assert bridge_trade.pair.is_cctp_bridge()
    assert bridge_trade.planned_quantity == Decimal(-5000)

    # 6. Assert the bridge position is reused
    bridge_trade_position = [
        p for p in state.portfolio.open_positions.values()
        if p.pair.is_cctp_bridge()
    ][0]
    assert bridge_trade in bridge_trade_position.trades.values()


def test_net_flow_same_chain_cancels(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair_base: TradingPairIdentifier,
    vault_pair_base: TradingPairIdentifier,
    vault_pair_base_2: TradingPairIdentifier,
):
    """Verify that net flow calculation cancels opposing trades on the same chain.

    When selling $5000 and buying $4000 on Base, only a $1000 bridge-back
    should be injected, not separate $5000 back + $4000 out.

    1. Create a state with reserves and an existing bridge position on Base
    2. Create a vault sell ($5000) and a vault buy ($4000) both on Base
    3. Call inject_cctp_bridge_trades
    4. Assert exactly one bridge trade was injected (the net $1000 bridge-back)
    5. Assert the bridge trade is a sell for $1000
    """
    # 1. Create state with bridge position on Base
    state = _make_state_with_reserves(usdc_arbitrum)

    # Create and complete a bridge-out to establish the bridge position
    _, bridge_out, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=cctp_pair_base,
        quantity=None,
        reserve=Decimal(10_000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )
    _execute_trade(state, bridge_out, TS, Decimal(10_000), Decimal(10_000))

    # Open vault position 1 (will sell)
    _, vault_buy_1, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=vault_pair_base,
        quantity=None,
        reserve=Decimal(5000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )
    _execute_trade(state, vault_buy_1, TS, Decimal(5000), Decimal(5000))

    universe = _make_mock_universe([cctp_pair_base, vault_pair_base, vault_pair_base_2])

    # 2. Create a vault sell ($5000) on Base
    vault_pos_1 = [p for p in state.portfolio.open_positions.values() if p.pair == vault_pair_base][0]
    _, vault_sell, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=vault_pair_base,
        quantity=Decimal(-5000),
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
        position=vault_pos_1,
        closing=True,
    )

    # Create a vault buy ($4000) on Base (different vault)
    _, vault_buy_2, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=vault_pair_base_2,
        quantity=None,
        reserve=Decimal(4000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )

    # 3. Inject bridge trades
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[vault_sell, vault_buy_2],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    # 4. Assert only one bridge trade was injected (net $1000 bridge-back)
    bridge_trades = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_trades) == 1

    # 5. Assert the bridge trade is a sell for $1000
    bridge_trade = bridge_trades[0]
    assert bridge_trade.is_sell()
    assert bridge_trade.planned_quantity == Decimal("-1000")


def test_no_bridge_for_primary_chain(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair_base: TradingPairIdentifier,
    vault_pair_primary: TradingPairIdentifier,
):
    """Verify that trades on the primary chain do not trigger bridge injection.

    1. Create a state with reserves
    2. Create vault buy and sell trades on the primary chain (Arbitrum)
    3. Call inject_cctp_bridge_trades
    4. Assert no bridge trades were injected
    5. Assert the returned list equals the original trades
    """
    # 1. Create state
    state = _make_state_with_reserves(usdc_arbitrum)
    universe = _make_mock_universe([cctp_pair_base, vault_pair_primary])

    # 2. Create a vault buy on the primary chain
    _, vault_buy, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=vault_pair_primary,
        quantity=None,
        reserve=Decimal(5000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )

    # 3. Inject bridge trades
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[vault_buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    # 4. No bridge trades injected
    bridge_trades = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_trades) == 0

    # 5. The returned list is the same as the input
    assert result == [vault_buy]


def test_multiple_satellite_chains(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair_base: TradingPairIdentifier,
    cctp_pair_avalanche: TradingPairIdentifier,
    vault_pair_base: TradingPairIdentifier,
    vault_pair_avalanche: TradingPairIdentifier,
):
    """Verify correct bridge injection for trades on multiple satellite chains.

    1. Create a state with reserves
    2. Create a vault buy on Base ($3000) and a vault buy on Avalanche ($2000)
    3. Call inject_cctp_bridge_trades with both bridge pairs in the universe
    4. Assert two bridge trades were injected (one per satellite chain)
    5. Assert the Base bridge-out is for $3000 and the Avalanche bridge-out is for $2000
    """
    # 1. Create state
    state = _make_state_with_reserves(usdc_arbitrum)
    universe = _make_mock_universe([
        cctp_pair_base,
        cctp_pair_avalanche,
        vault_pair_base,
        vault_pair_avalanche,
    ])

    # 2. Create vault buys on two satellite chains
    _, base_buy, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=vault_pair_base,
        quantity=None,
        reserve=Decimal(3000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )

    _, avax_buy, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=vault_pair_avalanche,
        quantity=None,
        reserve=Decimal(2000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )

    # 3. Inject bridge trades
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[base_buy, avax_buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    # 4. Assert two bridge trades were injected
    bridge_trades = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_trades) == 2

    # 5. Assert correct amounts per chain
    base_bridge = [t for t in bridge_trades if t.pair.get_destination_chain_id() == BASE_CHAIN_ID]
    avax_bridge = [t for t in bridge_trades if t.pair.get_destination_chain_id() == AVALANCHE_CHAIN_ID]

    assert len(base_bridge) == 1
    assert len(avax_bridge) == 1

    assert base_bridge[0].is_buy()
    assert base_bridge[0].planned_reserve == Decimal("3000")

    assert avax_bridge[0].is_buy()
    assert avax_bridge[0].planned_reserve == Decimal("2000")


@pytest.fixture()
def vault_pair_hypercore(usdc_arbitrum: AssetIdentifier) -> TradingPairIdentifier:
    """A HyperCore vault pair (chain_id 9999)."""
    vault_token = AssetIdentifier(
        chain_id=HYPERCORE_CHAIN_ID,
        address="0x0000000000000000000000000000000000000abc",
        token_symbol="HCvault",
        decimals=6,
    )
    return TradingPairIdentifier(
        base=vault_token,
        quote=AssetIdentifier(
            chain_id=HYPERCORE_CHAIN_ID,
            address="0x0000000000000000000000000000000000000def",
            token_symbol="USDC",
            decimals=6,
        ),
        pool_address="0x0000000000000000000000000000000000000aaa",
        exchange_address="0x0000000000000000000000000000000000000bbb",
        fee=0,
        kind=TradingPairKind.vault,
    )


def test_no_bridge_for_hypercore_vaults(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair_base: TradingPairIdentifier,
    vault_pair_hypercore: TradingPairIdentifier,
):
    """Verify that HyperCore vault trades do not trigger CCTP bridge injection.

    HyperCore vaults (chain_id 9999) have their own multi-phase settlement
    mechanism and do not need CCTP bridging.

    1. Create a state with reserves
    2. Create a HyperCore vault buy trade (chain_id 9999)
    3. Call inject_cctp_bridge_trades
    4. Assert no bridge trades were injected
    """
    # 1. Create state
    state = _make_state_with_reserves(usdc_arbitrum)
    universe = _make_mock_universe([cctp_pair_base, vault_pair_hypercore])

    # 2. Create a HyperCore vault buy
    _, vault_buy, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=vault_pair_hypercore,
        quantity=None,
        reserve=Decimal(5000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )

    # 3. Inject bridge trades
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[vault_buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    # 4. No bridge trades injected — HyperCore handles its own settlement
    bridge_trades = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_trades) == 0
    assert result == [vault_buy]
