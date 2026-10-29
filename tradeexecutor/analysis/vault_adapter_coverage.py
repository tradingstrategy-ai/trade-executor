"""Offline evidence inventory for the recorded NB05 vault universe.

This module never constructs a vault, reads an RPC endpoint, or certifies live
availability. Adapter names come from the selected eth-defi factory's source,
preserving its feature precedence without running notebook or adapter code.
"""

import ast
import collections
import re

from tradeexecutor.curator.curator import EXCLUDED_VAULTS


LIGHTER_LLP_KEY = (9998, "lighter-pool-281474976710654")

# These are related tests, not evidence that each deployment completed a lifecycle.
RELATED_TESTS = {
    "erc4626": ["tests/erc_4626/test_vault_trading_e2e.py"],
    "async": ["tests/ethereum/test_vault_settlement_lifecycle.py"],
    "lagoon": ["tests/erc_4626/test_vault_async_lagoon_erc_7540.py"],
    "hypercore": ["tests/ethereum/test_lagoon_crosschain_hypercore_simulated.py"],
    "lighter": ["tests/lagoon/test_lagoon_lighter_test_trade.py"],
}


def select_nb05_sources(notebook: dict, metadata: dict) -> set[tuple[int, str]]:
    """Read the recorded parent list and apply NB05's tagged-native union.

    Only the literal parent assignment is inspected. No notebook cell executes.
    Selection deliberately preserves parent membership instead of re-screening
    lifetime returns. The default LLP and private-access exclusions match NB05.
    """
    assignments = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source = source if isinstance(source, str) else "".join(source)
        if "NB04_SOURCE_VAULTS =" not in source:
            continue
        assignments.extend(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "NB04_SOURCE_VAULTS"
                for t in node.targets
            )
        )
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.List):
        raise ValueError("Expected one literal NB04_SOURCE_VAULTS assignment")

    parent = []
    for item in assignments[0].value.elts:
        if not (
            isinstance(item, ast.Tuple)
            and len(item.elts) == 2
            and isinstance(item.elts[0], ast.Call)
            and isinstance(item.elts[0].func, ast.Name)
            and item.elts[0].func.id == "ChainId"
            and len(item.elts[0].args) == 1
            and not item.elts[0].keywords
        ):
            raise ValueError(
                "Parent identities must be (ChainId(integer), literal address)"
            )
        chain = ast.literal_eval(item.elts[0].args[0])
        address = ast.literal_eval(item.elts[1])
        if type(chain) is not int or not isinstance(address, str):
            raise ValueError("Invalid parent identity")
        parent.append((chain, address.lower()))
    if len(set(parent)) != len(parent):
        raise ValueError("Duplicate parent identity")

    selected = set(parent) - {LIGHTER_LLP_KEY}
    restricted = {"private", "whitelisted", "permissioned", "restricted"}
    for entry in metadata["vaults"]:
        if "high_frequency_high_win_rate" not in (entry.get("strategy_tags") or []):
            continue
        protocol = entry.get("protocol_slug")
        if entry["chain_id"] != {"lighter": 9998, "hyperliquid": 9999}.get(protocol):
            continue
        address = entry["address"].lower()
        flags = {str(f).casefold().replace("-", "_") for f in entry.get("flags") or []}
        if (
            entry.get("normalised_denomination") != "USDC"
            or address in EXCLUDED_VAULTS
            or flags
            & {"subvault", "private", "private_vault", "whitelisted", "permissioned"}
            or str(entry.get("deposit_permission") or "").casefold() in restricted
            or str((entry.get("whitelist") or {}).get("status") or "").casefold()
            in restricted
            or re.search(
                r"\bprivate\s+(?:lagoon\s+)?vault\b",
                entry.get("notes") or "",
                re.IGNORECASE,
            )
        ):
            continue
        selected.add((entry["chain_id"], address))
    return selected - {LIGHTER_LLP_KEY}


def _matches_features(node: ast.expr, features: set[str]) -> bool:
    """Evaluate only feature-membership predicates, never arbitrary Python."""
    if isinstance(node, ast.BoolOp):
        values = [_matches_features(value, features) for value in node.values]
        if isinstance(node.op, ast.Or):
            return any(values)
        if isinstance(node.op, ast.And):
            return all(values)
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.In)
        and isinstance(node.left, ast.Attribute)
        and isinstance(node.left.value, ast.Name)
        and node.left.value.id == "ERC4626Feature"
        and isinstance(node.comparators[0], ast.Name)
        and node.comparators[0].id == "features"
    ):
        return node.left.attr in features
    raise ValueError(f"Unsupported factory predicate: {ast.unparse(node)}")


def resolve_adapter_class(classification_source: str, features: set[str]) -> str | None:
    """Resolve the factory-selected class statically, including overlapping features.

    Fail explicitly if the factory changes shape. Constructors (notably Lagoon's)
    can read chain state, so importing and constructing adapters is inappropriate
    for an offline inventory.
    """
    factory = next(
        (
            node
            for node in ast.parse(classification_source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "create_vault_instance"
        ),
        None,
    )
    if factory is None:
        raise ValueError("Missing create_vault_instance factory")
    branch = next(
        (
            node
            for node in factory.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Attribute)
            and isinstance(node.test.left.value, ast.Name)
            and node.test.left.value.id == "ERC4626Feature"
        ),
        None,
    )
    if branch is None:
        raise ValueError("Factory has no feature dispatch")
    while True:
        if _matches_features(branch.test, features):
            body = branch.body
            break
        if len(branch.orelse) == 1 and isinstance(branch.orelse[0], ast.If):
            branch = branch.orelse[0]
        else:
            body = branch.orelse
            break
    imports = {
        alias.asname or alias.name: f"{node.module}.{alias.name}"
        for node in body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    for node in body:
        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Constant) and node.value.value is None:
                return None
            if isinstance(node.value, ast.Call) and isinstance(
                node.value.func, ast.Name
            ):
                return imports[node.value.func.id]
    raise ValueError("Factory branch has no statically resolvable adapter class")


def _saved_trade_counts(state: dict | None) -> collections.Counter:
    """Count successful vault trades across all saved position collections once."""
    counts = collections.Counter()
    seen = set()
    for collection in (state or {}).get("portfolio", {}).values():
        if not isinstance(collection, dict):
            continue
        for position in collection.values():
            if not isinstance(position, dict) or "trades" not in position:
                continue
            for trade in position["trades"].values():
                if trade["trade_id"] in seen:
                    continue
                seen.add(trade["trade_id"])
                pair = trade["pair"]
                if (
                    pair["kind"] == "vault"
                    and trade.get("executed_at") is not None
                    and trade.get("failed_at") is None
                ):
                    counts[
                        (pair["base"]["chain_id"], pair["pool_address"].lower())
                    ] += 1
    return counts


def build_vault_adapter_coverage(
    source_keys: set[tuple[int, str]],
    metadata: dict,
    classification_source: str,
    *,
    state: dict | None = None,
    historical_reports: list[tuple[str, dict]] | None = None,
) -> list[dict]:
    """Build one evidence record per identity without asserting live readiness.

    Capability and admission fields are dated input observations. Test references
    describe related family coverage, never exact-deployment certification.
    """
    by_key = {}
    for entry in metadata["vaults"]:
        key = (entry["chain_id"], entry["address"].lower())
        if key in by_key:
            raise ValueError(f"Duplicate metadata identity: {key}")
        by_key[key] = entry
    missing = source_keys - by_key.keys()
    if missing:
        raise ValueError(f"Source identities missing from metadata: {sorted(missing)}")
    counts = _saved_trade_counts(state)
    unknown_trades = counts.keys() - source_keys
    if unknown_trades:
        raise ValueError(
            f"Saved vault trades outside source universe: {sorted(unknown_trades)}"
        )

    historical = {}
    for name, report in historical_reports or []:
        for result in report["results"]:
            # Later amendment reports replace the earlier attempt for this identity.
            historical[result["vault_id"].lower()] = {
                "report": name,
                "recorded_outcome": result["attempt"].get("result"),
                "recorded_status": result.get("row", {}).get("status"),
                "operation": result["attempt"].get("operation"),
                "detail": result["attempt"].get("detail"),
                "trade_executor_commit": report["run"].get("trade_executor_commit"),
                "eth_defi_commit": report["run"].get("eth_defi_commit"),
                "fork_blocks": report["run"].get("fork_blocks"),
            }

    rows = []
    for chain, address in sorted(source_keys):
        entry = by_key[chain, address]
        features = set(entry.get("features") or [])
        capability = entry.get("deposit_manager")
        custody_chain = {9998: 1, 9999: 999}.get(chain, chain)
        gaps = []
        tests = list(RELATED_TESTS["erc4626"])
        guard = (
            "manager-selected selectors, approval spender, custody owner and receiver"
        )
        route = "vault"
        adapter = None
        if chain == 9998:
            route = "missing_lighter_pool_route"
            tests = list(RELATED_TESTS["lighter"])
            guard = "Ethereum Lighter custody policy; pool selection is API-key software policy"
            gaps.extend(
                [
                    "lighter_pool_route",
                    "native_operation_persistence",
                    "native_nav_conservation",
                    "lighter_deployment_mapping",
                ]
            )
        elif chain == 9999:
            route = "hypercore_vault"
            adapter = "eth_defi.hyperliquid.vault.HyperliquidVault"
            tests = list(RELATED_TESTS["hypercore"])
            guard = "HyperEVM custody and explicit Hypercore vault whitelist"
            gaps.extend(
                ["native_satellite_funding_evidence", "native_fee_and_exit_evidence"]
            )
        else:
            adapter = resolve_adapter_class(classification_source, features)
            if adapter is None:
                gaps.append("adapter_class_unsupported")

        if "atoma_like" in features:
            gaps.extend(
                [
                    "atoma_request_receipt_parsing",
                    "zero_proceeds_redemption",
                    "atoma_claim_valuation",
                    "atoma_aggregate_recovery",
                ]
            )
            guard = "whitelistAtoma with compatible bytecode; remove obsolete broad permissions"
        if "gmx_gm" in features or "gmx_glv" in features:
            gaps.append("gmx_liquidity_transaction_adapter")
        if "symbiotic_like" in features:
            gaps.append("symbiotic_withdrawal_lifecycle")
        if (
            features & {"morpho_like", "morpho_v2_like"}
            and adapter
            and ".morpho." in adapter
        ):
            gaps.append("morpho_redemption_certification")
        if adapter and adapter.endswith("YearnCompounderVault"):
            gaps.append("yearn_compounder_certification")
        delayed_features = {"gains_like", "ember_like", "upshift_multi_asset_like"}
        if features & delayed_features:
            gaps.append("early_delayed_redemption_classification")
        if "d2_like" in features:
            gaps.append("d2_actual_phase_and_eligibility_evidence")
        if (capability or {}).get("redemption_flow") == "asynchronous" or features & {
            "erc_7540_like",
            "gains_like",
            "ember_like",
            "atoma_like",
        }:
            tests.extend(RELATED_TESTS["async"])
        if "lagoon_like" in features:
            tests.extend(RELATED_TESTS["lagoon"])

        assets = list((capability or {}).get("deposit_assets") or [])
        denomination = entry.get("denomination_token_address")
        if assets and (
            not denomination or denomination.lower() not in {a.lower() for a in assets}
        ):
            gaps.append("non_reserve_deposit_asset_conversion")
        # A positive static capability says nothing about today's account capacity.
        gaps.append("exact_deployment_guarded_lifecycle_evidence")
        rows.append(
            {
                "vault_id": f"{chain}-{address}",
                "name": entry["name"],
                "protocol_slug": entry.get("protocol_slug"),
                "dataset_chain_id": chain,
                "custody_chain_id": custody_chain,
                "signing_chain_id": 304 if chain == 9998 else custody_chain,
                "address": address,
                "features": sorted(features),
                "factory_selected_adapter_class": adapter
                if chain not in {9998, 9999}
                else None,
                "native_reader_class": adapter if chain == 9999 else None,
                "native_reader_module": {
                    9998: "eth_defi.lighter.vault",
                    9999: "eth_defi.hyperliquid.vault",
                }.get(chain),
                "executor_route": route,
                "recorded_manager_capability": capability,
                "recorded_deposit_assets": assets,
                "recorded_denomination_token": denomination,
                "recorded_deposits_open": entry.get("deposits_open"),
                "recorded_deposit_permission": entry.get("deposit_permission"),
                "guard_requirements": guard,
                "saved_successful_vault_trades": counts[chain, address]
                if state is not None
                else None,
                "related_executor_tests": sorted(set(tests)),
                "historical_exact_vault_attempt": historical.get(f"{chain}-{address}"),
                "required_execution_surfaces": [
                    "perform-test-trade",
                    "decide_trades/TradeExecution",
                    "lagoon-deploy-vault",
                ],
                "missing_evidence_or_implementation": gaps,
                "deployment_certified": False,
            }
        )
    return rows


def summarise_coverage(rows: list[dict]) -> dict:
    """Reconcile the complete population, routes, gaps and saved trade counts."""
    counts = [row["saved_successful_vault_trades"] for row in rows]
    return {
        "vaults": len(rows),
        "protocol_slugs": dict(
            sorted(collections.Counter(row["protocol_slug"] for row in rows).items())
        ),
        "dataset_chains": dict(
            sorted(
                collections.Counter(
                    str(row["dataset_chain_id"]) for row in rows
                ).items()
            )
        ),
        "factory_selected_adapter_classes": dict(
            sorted(
                collections.Counter(
                    row["factory_selected_adapter_class"] or "no_vault_class"
                    for row in rows
                ).items()
            )
        ),
        "native_reader_classes": dict(
            sorted(
                collections.Counter(
                    row["native_reader_class"]
                    for row in rows
                    if row["native_reader_class"]
                ).items()
            )
        ),
        "gaps": dict(
            sorted(
                collections.Counter(
                    gap
                    for row in rows
                    for gap in row["missing_evidence_or_implementation"]
                ).items()
            )
        ),
        "saved_traded_vaults": sum(count > 0 for count in counts)
        if all(count is not None for count in counts)
        else None,
        "saved_successful_vault_trades": sum(counts)
        if all(count is not None for count in counts)
        else None,
        "vaults_with_historical_attempt": sum(
            row["historical_exact_vault_attempt"] is not None for row in rows
        ),
        "new_live_lifecycle_tests_executed": False,
        "funded_transactions_executed": False,
    }
