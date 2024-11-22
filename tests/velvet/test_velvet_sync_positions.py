"""Test Velvet sync model."""
import datetime

import pytest

from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.velvet.vault import VelvetVaultSyncModel
from tradeexecutor.state.identifier import AssetIdentifier

from tradeexecutor.state.state import State
from tradingstrategy.pair import PandasPairUniverse



def test_velvet_sync_positions(
    base_example_vault: VelvetVault,
    base_usdc: AssetIdentifier,
    velvet_test_vault_pair_universe: PandasPairUniverse,
):
    """Sync velvet open positions

    - Do initial deposit scan.

    - Capture the initial USDC in the vault as a treasury

    - Capture DogMeIn open position
    """

    pair_universe = velvet_test_vault_pair_universe
    assert pair_universe.get_count() == 1

    sync_model = VelvetVaultSyncModel(
        vault=base_example_vault,
        hot_wallet=None,
    )

    state = State()
    portfolio = state.portfolio

    # Sync USDC
    sync_model.sync_initial(
        state,
        reserve_asset=base_usdc,
        reserve_token_price=1.0,
    )
    cycle = datetime.datetime.utcnow()
    sync_model.sync_treasury(cycle, state)
    assert portfolio.get_cash() == pytest.approx(2.674828)

    # Sync DogInMe
    sync_model.sync_positions(
        datetime.datetime.utcnow(),
        state,
        pair_universe=pair_universe,
    )