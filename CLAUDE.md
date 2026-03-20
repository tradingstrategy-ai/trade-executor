# Instructions to work with the code base

## English

- Use UK/British English instead of US English
- Say things like `visualise` instead of `visualize`
- For headings, only capitalise the first letter of heading, do not use title case

## Running notebooks

You can run a notebook from the command line using `jupyter execute`.
This supports multiprocessing (unlike `ipython` which forces single-process execution).

```shell
poetry run jupyter execute my-notebook.ipynb --inplace --timeout=900
```

- `--inplace` overwrites the notebook with executed results (cell outputs)
- `--timeout=900` sets a 15 minute per-cell execution timeout (use `-1` to disable for long-running optimisers)

Never use `ipython` command as it does not work with multiprocessing.

Alternative if you have IDE access, you can use the IDE to run the notebook.

## Running Python scripts

When running a Python script use `poetry run python` command instead of plain `python` command, so that the virtual environment is activated.

```shell
poetry run python scripts/logos/post-process-logo.py
```

## Running tests

If we have not run tests before make sure the user has created a gitignored file `.local-test.env` in the repository root. This will use `source` shell command to include the actual test secrets which lie outside the repository structure. Note: this file does not contain actual environment variables, just a `source` command to get them from elsewhere. **Never edit this file** and always ask the user to prepare the file for Claude Code.

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

## Pull requests

- Pull request description must have sections Why (the rational of change), Lessons learnt (memory) and Summary (what was changed). No test plan or verification section.
- Only push changes to remote when asked, never update pull requess automatically.
- Never push directly to a master if not told explicitly
- If the user ask to open a pull request as feature then start the PR title with "feat:" prefix and also add one line about the feature into `CHANGELOG.md`
- Each changelog entry should follow the date of the PR in YYYY-MM-DD format. Example: Something was updated (2026-01-01).
- Before opening or updating a pull request, format the code
- When merging pull request, squash and merge commits and use the PR description as the commit message
- If continuous integration (CI) tests fail on your PR, and they are marked flaky, run tests locally to repeat the issue if it is real flakiness or regression

### datetime

- Use naive UTC datetimes everywhere
- When using datetime class use `import datetime.datetime` and use `datetime.datetime` and `datetime.timedelta` as type hints
- Instead of `datetime.datetime.utcnow()` use `native_datetime_utc_now()` that is compatible across Python versions

### Python

- Always use module level imports, unless there are circular dependencies

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

## Web Fetching and 403

When fetching web pages, if `web_fetch` returns a 403 error, retry the request using the Chrome MCP tool to load the page in a real browser instead.

Prerequisites:

1. **Claude in Chrome extension** (v1.0.36+) - [Chrome Web Store](https://chromewebstore.google.com/detail/claude/fcoeoabgfenejglbffodgkkbkcdhcgfn)
2. **Google Chrome** running
3. **Direct Anthropic plan** (Pro, Max, Team, or Enterprise)


Browser tools are automatically available when the Chrome extension is connected. Use `@browser` in your Visual Studio Code prompt to activate the connection.

When using browser tools, Claude may ask for permission to visit specific domains. **Approve these prompts** to allow browser automation. You can also pre-approve domains in the Chrome extension settings.


