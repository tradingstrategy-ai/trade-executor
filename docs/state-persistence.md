# State persistence

The existing executor state mechanism is the main mechanism for all application
data, transactions, journaling, checkpoints and recovery progress. Use it for
strategies, protocol adapters, CLI commands and manual integration scripts.

**Ask for and obtain explicit permission from the user before building any
additional system for these purposes.** This includes separate transaction
journals, checkpoint files, recovery databases, sidecar state files and parallel
persistence or replay frameworks. A request to implement an integration or make
it resumable does not by itself authorise an additional state system.

## Use the existing mechanism

- Store execution data in [State](../tradeexecutor/state/state.py) and its
  existing portfolio, position, [TradeExecution](../tradeexecutor/state/trade.py)
  and [BlockchainTransaction](../tradeexecutor/state/blockhain_transaction.py)
  records. Extend the appropriate state model when new protocol data is needed.
- Load and save through the existing
  [StateStore](../tradeexecutor/state/store.py), using `store.sync(state)` and
  the existing persistence callbacks. A checkpoint is a save of this state.
- Reuse the execution model's
  [pre-broadcast callback](../tradeexecutor/strategy/execution_model.py)
  when a command needs to save prepared transactions before submission.
- Reuse existing settlement, reconciliation and repair paths. If they lack a
  capability, address that gap within the existing mechanism. Do not introduce
  a separate recovery system for one command or protocol.
- Derive progress and retry decisions from existing state and verified execution
  evidence. Logs, reports and exported diagnostics must not become a second
  authoritative execution ledger.

If an additional system appears necessary, explain the specific gap, why the
existing mechanism cannot cover it, and the proposed alternative to the user.
Obtain their explicit approval before implementing it. Existing specialised
artefacts are not permission to create more; this rule does not require migrating
them as part of an unrelated task.

See [CLI command patterns](../.claude/docs/cli-command.md) for state-store wiring
and [the production maintenance runbook](docker.md) for authoritative state
files, backups and stopping concurrent writers.
