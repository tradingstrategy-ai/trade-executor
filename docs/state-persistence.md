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
- Bind that callback before startup settlement for every live vault. A
  conditional protocol predicate leaves other startup claims without their
  exact signed transaction saved before broadcast. Resolve pending vault
  trades before valuation and treasury sync through the shared execution hook.
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

## Exclusive writer ownership

State-changing CLI commands acquire a non-blocking kernel lock before loading
their working state, making backups or submitting transactions. `start` holds
ownership for its entire lifetime, including when serving crash status. A
competing writer exits with status 1 and identifies the state and lock paths.
Read-only commands and non-writing previews/simulations can still read snapshots.
`console` holds writer ownership for the whole session because it exposes the
writable state store. Use `show-positions` or `show-valuation` for inspection
while the executor is running.

The lock lives beside the canonical state path, for example
`state/hyper-ai.json.lock`. File existence does not mean the lock is held: the
kernel releases ownership on exit or process death, including `SIGKILL`. Forked
workers close inherited handles without unlocking their parent. **Never delete
or replace lock files while processes may be running.** An old unlocked file is
safe to reuse; deleting it could let two writers lock different inodes.

New state-changing commands must use `with_state_lock()` from
`tradeexecutor.cli.state_lock`, inside the shared RPC-option decorator when
present. Declare genuinely non-writing modes with its `read_only` predicate.
The wrapper supplies a canonical `state_file` and holds ownership through command
cleanup. Helpers and `backup_state()` reuse the caller's ownership and must not
reacquire it. Scripts use `locked_state_file()` from `tradeexecutor.cli.bootstrap`
around the complete read/modify/write operation, using its yielded path.

These are advisory locks on Linux filesystems supporting `flock`. All writers
must participate and share the same state-directory mount and compatible file
permissions. Restart older daemons with the updated image before relying on
protection. Separate state copies or wallets shared by different state files
are not coordinated by this lock. State serialisation and atomic saves are
unchanged.

See [CLI command patterns](../.claude/docs/cli-command.md) for state-store wiring
and [the production maintenance runbook](docker.md) for authoritative state
files, backups and stopping concurrent writers.
