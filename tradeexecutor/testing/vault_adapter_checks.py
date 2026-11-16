"""Read-only probes for known NB05 executor integration boundaries.

These use deliberately minimal in-memory inputs, not protocol receipts. Results
report observed behaviour without asserting that a particular defect persists.
"""

from decimal import Decimal
from types import SimpleNamespace

from eth_defi.erc_4626.core import ERC4626Feature

from tradeexecutor.ethereum.lagoon.universe_config import (
    _collect_erc4626_vault_addresses,
    normalise_deployment_chain_id,
)
from tradeexecutor.ethereum.vault.vault_routing import convert_vault_flow_analysis
from tradeexecutor.state.identifier import DELAYED_VAULT_REDEMPTION_FEATURES


def probe_nb05_adapter_boundaries() -> dict:
    """Observe deployment mapping, payout conversion and early feature classification.

    No signer, RPC connection, vault constructor or broadcast is involved. The
    minimal pair and receipt-analysis objects isolate the interfaces under review.
    """
    pair = SimpleNamespace(
        is_vault=lambda: True,
        is_hyperliquid_vault=lambda: False,
        base=SimpleNamespace(chain_id=9998),
        pool_address="lighter-pool-281474976552443",
    )
    universe = SimpleNamespace(iterate_pairs=lambda: iter([pair]))
    result = {"lighter_deployment_chain": normalise_deployment_chain_id(9998)}
    try:
        _collect_erc4626_vault_addresses(universe, {9998})
        result["lighter_erc4626_collection"] = {"exception": None}
    except (ValueError, TypeError, AssertionError) as error:
        result["lighter_erc4626_collection"] = {
            "exception": type(error).__name__,
            "message": str(error),
        }
    try:
        amounts = convert_vault_flow_analysis(
            SimpleNamespace(denomination_amount=Decimal(0), share_count=Decimal(1)),
            direction="redeem",
        )
        result["zero_proceeds_conversion"] = {
            "exception": None,
            "amounts": [str(value) for value in amounts],
        }
    except (ValueError, TypeError, AssertionError) as error:
        result["zero_proceeds_conversion"] = {
            "exception": type(error).__name__,
            "message": str(error),
        }
    result["early_delayed_redemption_features"] = {
        name: ERC4626Feature(name) in DELAYED_VAULT_REDEMPTION_FEATURES
        for name in (
            "gains_like",
            "upshift_like",
            "upshift_multi_asset_like",
            "ember_like",
            "atoma_like",
            "lighter_native",
        )
    }
    return result
