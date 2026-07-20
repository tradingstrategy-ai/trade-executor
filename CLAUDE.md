# Instructions to work with the code base

## Reference docs for Claude

Topic deep-dives live under `.claude/docs/`. Consult the relevant one before
working on that area:

| Doc | Description |
|-----|-------------|
| `.claude/docs/agent-tricks-and-troubleshooting.md` | **MANDATORY read before ANY Claude CLI or Codex CLI invocation** (reviews, sanity checks, or one-off runs) |
| `.claude/docs/cli-command.md` | CLI command patterns for trade-executor |
| `.claude/docs/alpha-model.md` | Base synchronous `AlphaModel` — the signals → weights → targets → trades pipeline, risk caps, trade generation gates, diagnostics charts, with `hyper-ai.py` as the reference implementation |
| `.claude/docs/vault-deposit-redeem.md` | Synchronous and async (ERC-7540 / Lagoon / Ostium) vault deposit and redeem flows |
| `.claude/docs/hypercore-vault.md` | HyperCore native vault execution — properties, data structures, deposit/withdrawal phases, HyperEVM interactions, modules and diagrams |
| `.claude/docs/phase-aware-alpha-model.md` | Phase-aware alpha model — parking window-gated vault deposits in a yield-bearing queue vault, park/promote event log, correctness invariants, backtest window modelling, diagnostics charts |

## Agent review workflows

- **BLOCKING REQUIREMENT: before running ANY `codex` / `codex exec` or `claude` CLI command — reviews, sanity checks, or one-off runs — you MUST first read `.claude/docs/agent-tricks-and-troubleshooting.md`.** Do not invoke either CLI until you have read it in the current session. This is not optional and applies even when the call "looks trivial".
- Follow its recommended invocation patterns for plan reviews, code reviews, PR reviews, tool restrictions, timeouts, and handling silent or hanging agent runs.
- Always run non-interactive Codex reviews in **streaming mode** (`codex exec --json`) written to a raw file — never plain text mode piped through `tail`/`head`, which buffers output until completion and makes the run look hung.
- `codex exec` selects the sandbox directly (`--sandbox read-only` for reviews) and does **not** accept `--ask-for-approval` (that flag is interactive-only).
- Do not fall back to generic `claude --help`, plugin docs, or ad-hoc CLI flags until the local troubleshooting doc has been checked.

## English

- Use UK/British English instead of US English
- Say things like `visualise` instead of `visualize`
- For headings, only capitalise the first letter of heading, do not use title case

## Running notebooks

You can run a notebook from the command line using `jupyter execute`.
This supports multiprocessing (unlike `ipython` which forces single-process execution).

```shell
source .local-test.env && poetry run jupyter execute my-notebook.ipynb --inplace --timeout=900
```

- `--inplace` overwrites the notebook with executed results (cell outputs)
- `--timeout=900` sets a 15 minute per-cell execution timeout (use `-1` to disable for long-running optimisers)

Never use `ipython` command as it does not work with multiprocessing.

Alternative if you have IDE access, you can use the IDE to run the notebook.

## Running Python scripts

When running a Python script use `poetry run python` command instead of plain `python` command, so that the virtual environment is activated.

```shell
source .local-test.env && poetry run python scripts/logos/post-process-logo.py
```

## Running trade-executor

E.g. to test CLI commands

```shell
source .local-test.env && poetry run trade-executor --help
```

## Running tests

Settings are sourced from `.local-test.env` in the repository root. This will use `source` shell command to include the actual test secrets which lie outside the repository structure. Note: this file does not contain actual environment variables, just a `source` command to get them from elsewhere. **Never edit this file** and always ask the user to prepare the file for Claude Code.

Check that the environment is ready by running:

```shell
source .local-test.env
```

This should exit without errors.

To run tests you need to use the installed Poetry environment, with given environment secrets file.

To run tests use the `pytest` wrapper command:

```shell
source .local-test.env && poetry run pytest {test case name or pattern here}
```

Always prefix pytest command with relevant source command,
otherwise the test cannot find environment variables.

- Avoid running the whole test suite as it takes several minutes
- Only run specific test cases you need and if you need to run multiple tests, run them one by one to deal with timeout issues
- If you are running more than five tests in one command, you need to use `pytest -n auto` because the command will fail due to long test duration, use this also if you run a single test module with multiple slow test cases
- Use `pytest --log-cli-level=info` when diagnosing failingtests

Timeouts

- When running a single pytest or any test commands, always use an extended timeout
  by specifying `timeout: 180000` (3 minutes) in the bash tool parameters.
- When running multiple tests, specify `timeout: 360000` (6 minutes) in the bash tool parameters.

## Formatting code

Don't format code.

## Git worktrees

- When creating a git worktree, copy `.local-test.env` from the repo root.
- For worktrees, unless you are changing package dependencies, use `poetry run` from the parent repo virtualenv
- The editable install `.pth` file hardcodes the parent repo path, so Python imports resolve to parent repo source instead of worktree source. When running tests or scripts in a worktree, always prepend `PYTHONPATH="$(pwd):$PYTHONPATH"` so the worktree path takes priority:

```shell
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/my_test.py
```

- Never use `importlib` hacks in tests to work around this — use normal imports and set `PYTHONPATH` instead

## Pull requests

- For pull request, issue, CI, and merge work, use the `git` and `gh` command-line tools; no plugins are required.
- Pull request description must have sections Why (the rational of change), Lessons learnt (memory) and Summary (what was changed). No test plan or verification section. Use Markdown formatting, headings.
- When updating a pull request description, prefer `gh api` with the REST pull request endpoint instead of `gh pr edit --body`, because `gh pr edit` can fail with `GraphQL: Projects (classic) is being deprecated ... (repository.pullRequest.projectCards)`. Write longer descriptions to a temporary Markdown file and run `gh api repos/:owner/:repo/pulls/{pr-number} --method PATCH -F body=@/tmp/pr-body.md`.
- Only push changes to remote when asked, never update pull requess automatically.
- Never push directly to a master if not told explicitly
- If the user ask to open a pull request as feature then start the PR title with "feat:" prefix and also add one line about the feature into `CHANGELOG.md`
- Each changelog entry should follow the date of the PR in YYYY-MM-DD format. Example: Something was updated (2026-01-01).
- Before opening or updating a pull request, format the code
- When merging pull request, squash and merge commits and use the PR description as the commit message
- If continuous integration (CI) tests fail on your PR, and they are marked flaky, run tests locally to repeat the issue if it is real flakiness or regression
- If we add unrelated test fixes to the PR, include a section "Unrelated test fixes for CI green" in the PR description

### datetime and timestamps

- Use naive UTC datetimes everywhere: `datetime.datetime`, `pd.Timestamp` - never use timezones
- When using datetime class use `import datetime.datetime` and use `datetime.datetime` and `datetime.timedelta` as type hints
- Instead of `datetime.datetime.utcnow()` use `native_datetime_utc_now()` that is compatible across Python versions

### Python

- Always use global, level imports, unless facing circular import exception. NEVER USE FUNCTION LOCAL IMPORTS UNLESS TOLD SO OR ABSOLUTE NECESSARAY TO AVOID CIRCULAR IMPORTS.
- Using `assert` is ok, we never run with `python -O` 

### Enum

- For string enums, both members and values must in snake_case

### Pytest

- Never use test classes in pytest
- `pytest` tests should not have stdout output like `print`
- Use `pytest.approx()` to compare values of data and money `assert abs(aave_total_pnl - 96.6087) < 0.01` 
- Don't use logger.info() or logger.debug() inside test and fixture function bodies unless specifically asked
- Do not do excessive number of tests. Prefer one test for happy path and one test for bad path. Do several asserts within a single test case to have test coverage, but keeping the number of tests low.
- Always use pytest timeout and chat timeout when running tests. Use 5 minutes timeout unless you are running the full test suite.
- Akk tests must have docstring
- Docstring must stell what is being tested and why
- Docstring must have 1, 2, 3, N style ordered list of steps the test is taking, up to the hig level actions in the test. These steps must then repeat as line comments within the test body.
- If we mock something, we must describe why
- Have Python type hints for used pytest fixtures
- We cannot import from tests sub-tree: helper functions must go to live in `testing` submodules in the actual source tree
- Never set log level to `info` in pytest tests permanently, as it clogs CI output
- Don't write worktree path hacks into test - instead run Python using PYTHONPATH environment variable set to workaround worktree issues. Test and other Python modules should never contain refernces to worktrees.

### pyproject.toml

- When adding or updating dependencies in `pyproject.toml`, always add a comment why this dependency is needed for this project

## Python notebooks

- Whenever possible, prefer table output instead of print(). Use Pandas DataFrame and notebook's built-in display() function to render tabular data.

## Type Hints
- Always prefer native Python types over importing from typing module
- Use `dict`, `list`, `tuple`, `set` instead of `Dict`, `List`, `Tuple`, `Set`
- Use `type | None` instead of `Optional[type]`
- Use `str | int` instead of `Union[str, int]`
- Only import from typing when necessary (e.g., `Any`, `Callable`, `TypeVar`)

## ERC-20

- Don't do hardcoded token decimal multiply, use `TokenDetails.convert_to_raw()`
- Use `TokenDetails.transfer()` and similar - do not do raw ERC-20 contract calls unless needed
- Use `eth_defi.hotwallet.HotWallet` for deployer accounts and signing transactions when possible

## Web3

- Use create_multi_provider_web3() to create an RPC connection from JSON_xxx env vars
- Always use `create_multi_provider_web3()` for Web3 objects. Do not construct raw `Web3(HTTPProvider(...))` instances in code or tests unless explicitly requested by the user for a narrow debugging experiment. The eth-defi helper installs required middleware, including cached `eth_chainId` reads, retry/fallback behaviour, Anvil metadata handling, and chain-specific middleware.
- When waiting for a transaction and then making smart contract calls that depend on its state changes, always use [`wait_for_transaction_receipt_robust()`](deps/web3-ethereum-defi/eth_defi/provider/receipt.py) instead of `web3.eth.wait_for_transaction_receipt()`. The robust helper waits for the transaction state change to propagate across RPC nodes and avoids stale reads from load-balanced providers.

## Web Fetching and 403

When fetching web pages, if `web_fetch` returns a 403 error, retry the request using the Chrome MCP tool to load the page in a real browser instead.

Prerequisites:

1. **Claude in Chrome extension** (v1.0.36+) - [Chrome Web Store](https://chromewebstore.google.com/detail/claude/fcoeoabgfenejglbffodgkkbkcdhcgfn)
2. **Google Chrome** running
3. **Direct Anthropic plan** (Pro, Max, Team, or Enterprise)


Browser tools are automatically available when the Chrome extension is connected. Use `@browser` in your Visual Studio Code prompt to activate the connection.

When using browser tools, Claude may ask for permission to visit specific domains. **Approve these prompts** to allow browser automation. You can also pre-approve domains in the Chrome extension settings.
