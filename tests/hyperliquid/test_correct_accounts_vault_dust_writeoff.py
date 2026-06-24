"""Test correct-accounts writes off Hypercore vault dust instead of spamming closed positions.

Background: Hypercore vault withdrawals cannot fully exit. The protocol refuses exact
full withdrawals when NAV moves between planning and execution, so a small residual
(the withdrawal safety margin, ~1.5 USDC) is left behind on-chain. The original position
is marked closed at exit because the close epsilon tolerates the residual. However,
``create_missing_vault_positions`` runs on every correct-accounts cycle and used to
re-materialise that on-chain residual as a brand-new zero-quantity closed position on
every run, spamming the closed-positions list with hundreds of empty positions (observed
on the hyper-ai strategy). This is the dust write-off behaviour other trading pairs
already have.

This module verifies:

1. A vault with dust-level on-chain equity (below the close epsilon) and no open position
   is written off: no position is created or closed.
2. A vault with real on-chain equity (above the close epsilon) and no open position still
   gets an open position created, i.e. the write-off only affects dust.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

from tradeexecutor.ethereum.vault.hypercore_vault import create_hypercore_vault_pair
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.account_correction import create_missing_vault_positions
from tradeexecutor.strategy.dust import HYPERLIQUID_VAULT_CLOSE_EPSILON


def _build_state_and_universe() -> tuple[State, MagicMock, object]:
    """Build a minimal state plus a mocked universe exposing a single Hypercore vault pair.

    Returns the state, the mocked strategy universe, and the vault pair the mocked
    ``translate_trading_pair`` will resolve to.
    """
    reserve_asset = AssetIdentifier(
        chain_id=999,
        address="0xb88339cb7199b77e23db6e890353e22632ba630f",
        token_symbol="USDC",
        decimals=6,
    )
    pair = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address="0x1111111111111111111111111111111111111111",
    )

    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve_asset, Decimal("100"), "Initial reserve")

    # The universe only needs to yield one dex pair (a sentinel); translate_trading_pair
    # is patched to map it to our Hypercore vault pair.
    strategy_universe = MagicMock()
    strategy_universe.data_universe.pairs.iterate_pairs.return_value = [object()]
    strategy_universe.reserve_assets = [reserve_asset]

    return state, strategy_universe, pair


def test_vault_dust_is_written_off():
    """Dust-level vault equity with no open position is written off, not re-created as a closed position.

    1. Build a state with no positions and a universe with one Hypercore vault pair.
    2. Run create_missing_vault_positions() with on-chain equity below the close epsilon.
    3. Verify no trades were created and no positions exist (dust is simply ignored).
    4. Run it a second time (the residual stays on-chain) and verify it still does not
       accumulate phantom positions — the exact regression that spammed hyper-ai.
    """

    # 1. Build a state with no positions and a universe with one Hypercore vault pair.
    state, strategy_universe, pair = _build_state_and_universe()

    # 2. Run create_missing_vault_positions() with on-chain equity below the close epsilon.
    dust_equity = float(HYPERLIQUID_VAULT_CLOSE_EPSILON) - 0.50
    with patch(
        "tradeexecutor.strategy.account_correction.translate_trading_pair",
        return_value=pair,
    ):
        created_trades = create_missing_vault_positions(
            strategy_universe=strategy_universe,
            state=state,
            strategy_cycle_at=datetime.datetime(2026, 4, 16),
            vault_value_func=lambda p: Decimal(str(dust_equity)),
        )

        # 3. Verify no trades were created and no positions exist (dust is simply ignored).
        assert created_trades == []
        assert len(state.portfolio.open_positions) == 0
        assert len(state.portfolio.closed_positions) == 0

        # 4. Run a second cycle: the on-chain residual is still there, but we must not
        #    accumulate a new phantom closed position each run.
        created_trades_again = create_missing_vault_positions(
            strategy_universe=strategy_universe,
            state=state,
            strategy_cycle_at=datetime.datetime(2026, 4, 17),
            vault_value_func=lambda p: Decimal(str(dust_equity)),
        )
        assert created_trades_again == []
        assert len(state.portfolio.open_positions) == 0
        assert len(state.portfolio.closed_positions) == 0


def test_vault_real_equity_still_opens_position():
    """Above-dust vault equity with no open position still creates an open position.

    This guards that the dust write-off does not suppress genuine missing positions.

    1. Build a state with no positions and a universe with one Hypercore vault pair.
    2. Run create_missing_vault_positions() with on-chain equity above the close epsilon.
    3. Verify one open position was created and it was not auto-closed.
    """

    # 1. Build a state with no positions and a universe with one Hypercore vault pair.
    state, strategy_universe, pair = _build_state_and_universe()

    # 2. Run create_missing_vault_positions() with on-chain equity above the close epsilon.
    real_equity = float(HYPERLIQUID_VAULT_CLOSE_EPSILON) + 50.0
    with patch(
        "tradeexecutor.strategy.account_correction.translate_trading_pair",
        return_value=pair,
    ):
        created_trades = create_missing_vault_positions(
            strategy_universe=strategy_universe,
            state=state,
            strategy_cycle_at=datetime.datetime(2026, 4, 16),
            vault_value_func=lambda p: Decimal(str(real_equity)),
        )

    # 3. Verify one open position was created and it was not auto-closed.
    assert len(created_trades) == 1
    assert len(state.portfolio.open_positions) == 1
    assert len(state.portfolio.closed_positions) == 0
