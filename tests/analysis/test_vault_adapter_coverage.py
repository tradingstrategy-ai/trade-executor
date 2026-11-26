"""Offline NB05 inventory coverage, including precedence and incomplete evidence."""

import inspect
import json
from pathlib import Path

import pytest
from eth_defi.erc_4626.classification import create_vault_instance

from tradeexecutor.analysis.vault_adapter_coverage import (
    build_vault_adapter_coverage,
    resolve_adapter_class,
    select_nb05_sources,
    summarise_coverage,
)


def test_vault_adapter_inventory_preserves_selection_and_evidence() -> None:
    """Inventory distinct lifecycles without executing notebook or adapter code.

    1. Build parent, tagged-native, private and LLP metadata plus inert notebook code.
    2. Select sources and inventory them using the real eth-defi factory source.
    3. Check routing identities, feature precedence, funding gaps and evidence limits.
    """
    # 1. Build parent, tagged-native, private and LLP metadata plus inert notebook code.
    cases = [
        (42161, "atoma", ["atoma_like"]),
        (1, "morpho", ["morpho_like", "yearn_compounder_like"]),
        (1, "yearn", ["yearn_v3_like", "yearn_compounder_like"]),
        (1, "yearn", ["yearn_compounder_like"]),
        (42161, "gmx", ["gmx_gm"]),
        (1, "symbiotic", ["symbiotic_like"]),
        (1, "upshift", ["upshift_like", "upshift_multi_asset_like"]),
        (1, "upshift", ["upshift_like"]),
    ]
    entries = [
        {
            "chain_id": chain,
            "address": f"0x{index:040x}",
            "name": f"Vault {index}",
            "protocol_slug": protocol,
            "features": features,
        }
        for index, (chain, protocol, features) in enumerate(cases, start=1)
    ]
    entries[6]["deposit_manager"] = {
        "deposit_assets": ["0x00000000000000000000000000000000000000bb"],
        "redemption_flow": "asynchronous",
    }
    entries[6]["denomination_token_address"] = (
        "0x00000000000000000000000000000000000000aa"
    )
    parent = ",\n".join(
        f"(ChainId({e['chain_id']}), {e['address']!r})" for e in entries
    )
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "raise RuntimeError('Notebook execution is forbidden')\n",
                    f"NB04_SOURCE_VAULTS = [{parent}, (ChainId(9998), 'lighter-pool-281474976710654')]\n",
                ],
            }
        ]
    }
    entries.extend(
        [
            {
                "chain_id": chain,
                "address": address,
                "name": address,
                "protocol_slug": protocol,
                "features": [],
                "strategy_tags": ["high_frequency_high_win_rate"],
                "normalised_denomination": "USDC",
                "deposit_permission": permission,
            }
            for chain, address, protocol, permission in [
                (9998, "lighter-pool-123", "lighter", "permissionless"),
                (9998, "lighter-pool-281474976710654", "lighter", "permissionless"),
                (
                    9999,
                    "0x0000000000000000000000000000000000000099",
                    "hyperliquid",
                    "permissionless",
                ),
                (
                    9999,
                    "0x0000000000000000000000000000000000000088",
                    "hyperliquid",
                    "private",
                ),
            ]
        ]
    )

    # 2. Select sources and inventory them using the real eth-defi factory source.
    metadata = {"vaults": entries}
    keys = select_nb05_sources(notebook, metadata)
    trade = {
        "trade_id": 1,
        "executed_at": 1,
        "failed_at": None,
        "pair": {
            "kind": "vault",
            "base": {"chain_id": 42161},
            "pool_address": entries[0]["address"],
        },
    }
    position = {"trades": {"1": trade}}
    state = {
        "portfolio": {
            "open_positions": {"1": position},
            "closed_positions": {"1": position},
        }
    }
    historical = [
        (
            name,
            {
                "run": {},
                "results": [
                    {
                        "vault_id": f"42161-{entries[0]['address']}",
                        "attempt": {"result": outcome},
                        "row": {"status": status},
                    }
                ],
            },
        )
        for name, outcome, status in [
            ("original.json", None, "success (simulated)"),
            ("amendment.json", "deposit_closed", "deposit_closed"),
        ]
    ]
    rows = build_vault_adapter_coverage(
        keys,
        metadata,
        inspect.getsource(create_vault_instance),
        state=state,
        historical_reports=historical,
    )
    by_name = {row["name"]: row for row in rows}

    # 3. Check routing identities, feature precedence, funding gaps and evidence limits.
    assert len(rows) == 10
    assert (9998, "lighter-pool-281474976710654") not in keys
    assert not any(row["recorded_deposit_permission"] == "private" for row in rows)
    assert by_name["lighter-pool-123"]["custody_chain_id"] == 1
    assert by_name["lighter-pool-123"]["signing_chain_id"] == 304
    assert by_name["lighter-pool-123"]["executor_route"] == "missing_lighter_pool_route"
    assert by_name["Vault 2"]["factory_selected_adapter_class"].endswith(
        "MorphoV1Vault"
    )
    assert by_name["Vault 3"]["factory_selected_adapter_class"].endswith("YearnV3Vault")
    assert by_name["Vault 4"]["factory_selected_adapter_class"].endswith(
        "YearnCompounderVault"
    )
    assert (
        "gmx_liquidity_transaction_adapter"
        in by_name["Vault 5"]["missing_evidence_or_implementation"]
    )
    assert (
        "symbiotic_withdrawal_lifecycle"
        in by_name["Vault 6"]["missing_evidence_or_implementation"]
    )
    assert (
        "non_reserve_deposit_asset_conversion"
        in by_name["Vault 7"]["missing_evidence_or_implementation"]
    )
    assert (
        "early_delayed_redemption_classification"
        not in by_name["Vault 8"]["missing_evidence_or_implementation"]
    )
    assert all(row["deployment_certified"] is False for row in rows)
    assert summarise_coverage(rows)["saved_successful_vault_trades"] == 1
    assert (
        by_name["Vault 1"]["historical_exact_vault_attempt"]["report"]
        == "amendment.json"
    )
    assert (
        by_name["Vault 1"]["historical_exact_vault_attempt"]["recorded_status"]
        == "deposit_closed"
    )
    assert by_name["Vault 2"]["historical_exact_vault_attempt"] is None


def test_vault_adapter_inventory_rejects_incomplete_inputs() -> None:
    """Refuse missing identities and unsupported source shapes rather than drop vaults.

    1. Build a minimal metadata input and load the real factory source.
    2. Reject missing and duplicate identities and executable parent expressions.
    3. Reject a changed factory predicate instead of guessing an adapter class.
    """
    # 1. Build a minimal metadata input and load the real factory source.
    entry = {
        "chain_id": 1,
        "address": "0x0000000000000000000000000000000000000001",
        "name": "Vault",
    }
    source = inspect.getsource(create_vault_instance)
    keys = {(1, entry["address"])}

    # 2. Reject missing and duplicate identities and executable parent expressions.
    with pytest.raises(ValueError, match="missing from metadata"):
        build_vault_adapter_coverage(keys, {"vaults": []}, source)
    with pytest.raises(ValueError, match="Duplicate metadata"):
        build_vault_adapter_coverage(keys, {"vaults": [entry, entry]}, source)
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": "NB04_SOURCE_VAULTS = [(dangerous(), 'address')]",
            }
        ]
    }
    with pytest.raises(ValueError, match="Parent identities"):
        select_nb05_sources(notebook, {"vaults": []})

    # 3. Reject a changed factory predicate instead of guessing an adapter class.
    changed = "def create_vault_instance():\n    if ERC4626Feature.morpho_like == features:\n        return None\n"
    with pytest.raises(ValueError, match="Unsupported factory predicate"):
        resolve_adapter_class(changed, {"morpho_like"})


def test_nb05_recorded_inventory_reconciles_all_sources() -> None:
    """Check the committed full inventory without requiring licensed input datasets.

    1. Load the sanitised assessment generated from the recorded NB05 inputs.
    2. Reconcile every source, selected class, saved trade and evidence count.
    3. Check the recorded gaps and retain all three execution surfaces per vault.
    """
    # 1. Load the sanitised assessment generated from the recorded NB05 inputs.
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/reports/nb05-live-adapter-coverage.json"
    )
    report = json.loads(path.read_text())
    rows = report["vaults"]

    # 2. Reconcile every source, selected class, saved trade and evidence count.
    assert report["summary"] == summarise_coverage(rows)
    assert len(rows) == len({row["vault_id"] for row in rows}) == 347
    assert len(report["summary"]["protocol_slugs"]) == 20
    assert sum(report["summary"]["protocol_slugs"].values()) == 347
    assert report["summary"]["saved_traded_vaults"] == 54
    assert report["summary"]["saved_successful_vault_trades"] == 597
    assert report["summary"]["vaults_with_historical_attempt"] == 71

    # 3. Check the recorded gaps and retain all three execution surfaces per vault.
    gaps = report["summary"]["gaps"]
    assert gaps["lighter_pool_route"] == 6
    assert gaps["atoma_aggregate_recovery"] == 2
    assert gaps["non_reserve_deposit_asset_conversion"] == 3
    assert gaps["early_delayed_redemption_classification"] == 23
    assert all(len(row["required_execution_surfaces"]) == 3 for row in rows)
    assert all(row["deployment_certified"] is False for row in rows)
    assert report["summary"]["new_live_lifecycle_tests_executed"] is False
    assert report["summary"]["funded_transactions_executed"] is False
