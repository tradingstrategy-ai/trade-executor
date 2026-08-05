# Lagoon guard migration

`lagoon-deploy-vault --guard-only` deploys a replacement
`TradingStrategyModuleV0` but does not alter the Safe itself. The Safe owners
must execute the proposed Safe transactions in the deployment artefact before
the executor can use the replacement module.

## Complete a migration

1. Keep the generated deployment artefact. Its `guard_migration` section
   contains the old and new guard addresses, the Safe address, and the
   proposed Safe transactions.
2. Have the Safe owners execute the proposed transactions.
3. Set `VAULT_ADAPTER_ADDRESS` to the new guard address in the executor
   configuration.
4. Start the executor. It reads the Safe's enabled modules, saves the observed
   migration status in the strategy state, and then checks that every
   configured primary and satellite Safe has its configured module enabled.

Do not treat the deployment artefact's
`enabled_modules_at_deployment` value as confirmation that the migration has
finished: it is a snapshot taken before the Safe owners execute the upgrade.

## Read the saved state

The primary deployment's `guard_migration` entry stores the live Safe
observation separately from the deployment instructions:

| Field | Meaning |
| --- | --- |
| `enabled_modules_at_deployment` | Historical Safe module snapshot created while deploying the replacement guard. |
| `proposed_safe_transactions` | Historical transaction plan for Safe owners. |
| `currently_enabled_modules` | Modules read from the primary Safe at `observed_at`. |
| `status` | `completed`, `pending`, `both_enabled`, or `no_expected_guard_enabled`. |
| `manual_intervention_required` | `false` only when the new guard is enabled and the old guard is no longer enabled. |
| `observed_at` | UTC time of the successful Safe module read. |

`pending` means only the old guard is enabled. `both_enabled` and
`no_expected_guard_enabled` also need owner intervention. In all three cases,
use the addresses and transaction plan in the deployment artefact to inspect
and correct the Safe; do not edit the observed state manually.

If the Safe cannot be read during start-up, the executor logs that the
migration status could not be refreshed. Resolve the RPC issue before relying
on the saved observation.

## Start-up safety check

The executor checks `isModuleEnabled(VAULT_ADAPTER_ADDRESS)` on the primary
Lagoon Safe before it starts execution. It performs the same check for every
configured satellite Safe. A mismatch stops start-up before trades are sent.

This check is intentionally independent of the migration status: the state is
an operational record, while module enablement is the live on-chain safety
condition.
