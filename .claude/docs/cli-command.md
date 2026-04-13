# CLI command patterns

This note explains how CLI commands are built in `trade-executor`, which helper functions are considered the standard plumbing, and which common patterns new commands should follow.

## Where commands live

- Typer app root: `tradeexecutor/cli/commands/app.py`
- Command modules: `tradeexecutor/cli/commands/*.py`
- Main registration import list: `tradeexecutor/cli/main.py`
- Shared options: `tradeexecutor/cli/commands/shared_options.py`
- Shared bootstrap helpers: `tradeexecutor/cli/bootstrap.py`
- Logging setup: `tradeexecutor/cli/log.py`

To add a new command:

1. Create a module in `tradeexecutor/cli/commands/`
2. Define a function decorated with `@app.command()`
3. Import that module or command in `tradeexecutor/cli/main.py`

Minimal example:

```python
from .app import app
from . import shared_options


@app.command()
def my_command(
    id: str = shared_options.id,
):
    ...
```

## Common command shapes

There are three common command families in this repo.

### Local state commands

These work only on the state file and usually do not need RPC or dataset access.

Examples:

- `show_positions`
- `prune_state`
- `correct_history`
- `repair_hypercore_dust`

Typical pattern:

1. Accept `id`, optional `strategy_file`, and optional `state_file`
2. Resolve `id` with `prepare_executor_id()` if needed
3. Default `state_file` to `state/{id}.json`
4. Call `setup_logging()` early unless the command is intentionally print-only
5. Open the state via `create_state_store()` and assert it exists
6. If the command mutates state, take a backup with `backup_state()` before changing anything
7. Mutate or inspect state
8. Save with `store.sync(state)` if the command changes state
9. End with `logger.info("All ok")` if the command has a clear repair or verification outcome

Representative code:

- `tradeexecutor/cli/commands/show_positions.py`
- `tradeexecutor/cli/commands/prune.py`
- `tradeexecutor/cli/commands/correct_history.py`
- `tradeexecutor/cli/commands/repair_hypercore_dust.py`

### Live or chain-aware commands

These need JSON-RPC connections, strategy setup, and often cache paths.

Examples:

- `check_wallet`
- `correct_accounts`
- `close_position`
- `repair`
- `start`
- `lagoon_settle`

Typical pattern:

1. Decorate with `@shared_options.with_json_rpc_options()`
2. Accept `rpc_kwargs: dict | None = None` as an internal placeholder
3. Resolve `id` with `prepare_executor_id()`
4. Call `setup_logging()`
5. Prepare cache with `prepare_cache()` if datasets or token caches are needed
6. Build `web3config = create_web3_config(...)`
7. Choose the active chain with `web3config.choose_single_chain()` or `set_default_chain(...)`
8. Build sync model, execution model, client, universe, pricing, routing as needed

Representative code:

- `tradeexecutor/cli/commands/check_wallet.py`
- `tradeexecutor/cli/commands/correct_accounts.py`
- `tradeexecutor/cli/commands/close_position.py`
- `tradeexecutor/cli/commands/start.py`

### Hybrid commands

Some commands are mostly local-state commands but still construct strategy plumbing or universe data.

Examples:

- `backtest`
- `console`
- `trade_ui`
- `hyperliquid_cleanup` helper functions

These usually follow the live-command shape even if they are run manually.

## Critical options

### `id`

Shared definition:

- `tradeexecutor/cli/commands/shared_options.py`

Resolver:

- `tradeexecutor/cli/bootstrap.py::prepare_executor_id()`

Purpose:

- gives the executor a stable identity
- is used to derive the default state path
- is used to derive the default cache path
- is used in file logging and some web metadata

Rules:

1. If explicitly passed, use it
2. Otherwise infer it from `Path(strategy_file).stem`
3. If neither is available, raise

Default inference:

```python
id = prepare_executor_id(id, strategy_file)
state_file = Path(f"state/{id}.json")
cache_path = prepare_cache(id, cache_path)
```

If a command can operate from just a state file, `strategy_file` may be optional. If a command needs strategy code, `strategy_file` should be required.

### `state_file`

Shared definition:

- `tradeexecutor/cli/commands/shared_options.py::state_file`

Standard pattern:

```python
if not state_file:
    id = prepare_executor_id(id, strategy_file)
    state_file = Path(f"state/{id}.json")
```

Use `create_state_store(state_file)` to open it. Many commands then assert that the store is not pristine before proceeding.

For mutating local-state commands, a stronger operator-safe pattern is:

```python
store = create_state_store(Path(state_file))
assert not store.is_pristine(), f"State file does not exist: {state_file}"

store, state = backup_state(
    state_file,
    backup_suffix="my-command-backup",
    unit_testing=unit_testing,
)
```

This is now the preferred approach for repair-style commands because it gives both an explicit existence check and a recoverable backup before mutation.

### `cache_path`

Helper:

- `tradeexecutor/cli/bootstrap.py::prepare_cache()`

Purpose:

- dataset cache root for Trading Strategy client data
- standard place for token caches and downloaded artefacts
- writable-path preflight check

Default:

- normal runs: `cache/{executor_id}`
- unit tests: `/tmp/trading-strategy-tests`

Use it whenever the command constructs a Trading Strategy client, universe, token cache, or anything that downloads and persists data.

### `web3config`

Helper:

- `tradeexecutor/cli/bootstrap.py::create_web3_config()`

Purpose:

- normalises all `JSON_RPC_*` inputs
- creates a `Web3Config`
- supports simulation / Anvil flows
- can carry chain-selection and fork metadata

Standard pattern for RPC commands:

```python
@shared_options.with_json_rpc_options()
def my_command(
    ...,
    rpc_kwargs: dict | None = None,
):
    web3config = create_web3_config(
        **rpc_kwargs,
        unit_testing=unit_testing,
        simulate=simulate,
    )
    web3config.choose_single_chain()
```

Notes:

- `with_json_rpc_options()` injects the public `json_rpc_*` Typer options and assembles them into `rpc_kwargs`
- if the strategy expects a particular chain, commands may call `web3config.set_default_chain(...)` before `choose_single_chain()`
- if the command allocates resources, call `web3config.close()` when done

## Logging pattern

Logging is normally wired through:

- `tradeexecutor/cli/log.py::setup_logging()`

Standard call:

```python
logger = setup_logging(log_level=log_level)
```

What it does:

- configures the root logger
- installs coloured console logs
- quiets noisy subsystems like `web3`, `requests`, `urllib3`, `matplotlib`, and `graphql`
- supports the special `log_level == "disabled"` mode used by tests
- optionally enables an in-memory ring buffer for the web UI

Common logging practice in commands:

1. Call `setup_logging()` very early
2. Use `logger.info(...)` for progress and summary lines
3. Use `logger.error(...)` before exiting on failure
4. Log important inferred paths such as the resolved `state_file` or `cache_path`
5. Use `print(...)` only for intentionally user-facing tabular or ad-hoc console output in local tools

Examples:

- `check_wallet` uses structured progress logging and ends with `logger.info("All ok")`
- `correct_accounts` logs the correction summary and only prints `All ok` when the final verification passes
- `repair_hypercore_dust` logs the resolved state path, duplicate diagnostics, save step, and final `All ok`
- `show_positions` is intentionally print-heavy because it is a pure inspection command

## How `All ok` happens

There is no automatic success footer in the CLI framework. `All ok` is always emitted explicitly by the command itself.

Examples:

- `tradeexecutor/cli/commands/check_wallet.py`
- `tradeexecutor/cli/commands/correct_accounts.py`
- `tradeexecutor/cli/commands/close_all.py`
- `tradeexecutor/cli/commands/close_position.py`
- `tradeexecutor/cli/commands/check_universe.py`
- `tradeexecutor/cli/commands/perform_test_trade.py`

Typical pattern:

1. Do the work
2. Run a final validation or read-back check
3. If clean, log `All ok`
4. Exit with status `0`

In some commands the success path is:

```python
logger.info("All ok")
sys.exit(0)
```

In others it is simply:

```python
logger.info("All ok")
```

Use `All ok` when the command has a meaningful final pass/fail state, especially after verification against chain state or final saved state. Do not add it to every inspection-only command.

For purely informational commands, `All ok` is optional. For commands that repair, validate, or close out an operator workflow, it is strongly preferred.

## Shared option injection

For RPC commands, prefer:

```python
@shared_options.with_json_rpc_options()
def my_command(
    ...,
    rpc_kwargs: dict | None = None,
):
    ...
```

`with_json_rpc_options()`:

- rewrites the Typer signature to expose the shared `json_rpc_*` options
- collects them into one `rpc_kwargs` dictionary
- keeps the command body cleaner and consistent across the repo

This is the standard pattern for chain-aware commands. Do not manually duplicate all `json_rpc_*` options unless there is a very unusual reason.

## Common bootstrap sequence

For strategy-aware live commands, this is the usual sequence:

1. `id = prepare_executor_id(id, strategy_file)`
2. `logger = setup_logging(log_level)`
3. `mod = read_strategy_module(strategy_file)`
4. `cache_path = prepare_cache(id, cache_path, unit_testing=unit_testing)`
5. `web3config = create_web3_config(**rpc_kwargs, ...)`
6. `web3config.choose_single_chain()`
7. `execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_execution_and_sync_model(...)`
8. `client, routing_model = create_client(...)`
9. `store = create_state_store(state_file)`
10. load or create state
11. construct universe / routing / pricing if needed
12. perform command-specific action
13. `store.sync(state)` if state changed
14. final verification
15. `logger.info("All ok")` if appropriate

Examples:

- `close_position`
- `correct_accounts`
- `repair`
- `console`

## State-saving patterns

If a command mutates state:

- usually resolve and log the target path first
- use `create_state_store()` to confirm the state exists
- use `backup_state()` before saving if the mutation is operator-facing

Typical safe mutation pattern:

```python
store = create_state_store(Path(state_file))
assert not store.is_pristine()
store, state = backup_state(state_file, unit_testing=unit_testing)
...
store.sync(state)
```

This is common in repair and correction commands.

When choosing a backup suffix, prefer a command-specific name such as `correct-history-backup` or `repair-hypercore-dust-backup`, so operators can immediately see which tool created the snapshot.

## Exit and failure conventions

Common conventions:

- raise `RuntimeError` or `AssertionError` for unrecoverable operator errors
- use `sys.exit(0)` or `sys.exit(1)` in commands with explicit command-line success/failure semantics
- log before exiting so operators understand what happened

Examples:

- `correct_accounts` exits `0` on clean final verification and `1` on unclean final verification
- `check_accounts` and `check_wallet` follow a similar verification-first pattern
- local helper commands often just raise and let Typer surface the error

## Good templates to copy

Use these commands as templates depending on what you are building.

For local state inspection:

- `show_positions.py`

For local state mutation with backup:

- `prune.py`
- `correct_history.py`

For RPC and strategy-aware verification:

- `check_wallet.py`
- `check_accounts.py`

For RPC and state mutation:

- `correct_accounts.py`
- `close_position.py`
- `repair.py`

For local repair with duplicate diagnostics:

- `repair_hypercore_dust.py`

For long-running daemon entry:

- `start.py`
- `webapi.py`

## Checklist for new commands

Before opening a PR for a new CLI command, check:

1. Is the command registered through `tradeexecutor/cli/main.py`?
2. Does it use shared options instead of inventing new copies?
3. Does it resolve `id` with `prepare_executor_id()` if it needs default state or cache paths?
4. Does it use `prepare_cache()` if it downloads datasets or token metadata?
5. Does it use `@shared_options.with_json_rpc_options()` and `create_web3_config()` if it touches chain state?
6. Does it call `setup_logging()` early?
7. Does it save state with `store.sync()` only when intended?
8. Does it back up state first if the mutation is risky?
9. Does it provide a final success message only when there is a meaningful verified success state?
10. Does the command help output look like the other commands in the repo?

## Findings from review

Comparing `repair_hypercore_dust` against `prune_state`, `correct_history`, `check_wallet`, `correct_accounts`, and `close_position` reinforced a few practical rules:

- Local repair commands should still look like first-class operator commands, not one-off scripts.
- Inferring `state/{id}.json` and logging the resolved path makes Docker and ad-hoc operator use much safer.
- The normal mutation pattern is `create_state_store()` for existence checking, then `backup_state()` before edits, then `store.sync(state)`.
- Commands with a crisp success contract should end with `logger.info("All ok")`; operators look for that line.
- `strategy_file` can be optional for local-state-only commands, but `id` inference should still route through `prepare_executor_id()` when `state_file` is omitted.
