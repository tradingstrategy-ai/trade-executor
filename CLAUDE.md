# Instructions to work with the code base

## English

- Use UK/British English instead of US English
- Say things like `visualise` instead of `visualize`
- For headings, only capitalise the first letter of heading, do not use title case

## Running notebooks

You can test if a notebook runs from the command line with IPython command.

Example:

```shell
poetry run ipython my-notebook.ipynb
```

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
source .local-test.env && poetry pytest run {test case name or pattern here}
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

- Never push directly to a master, and open a pull request when asked.
- Do not include test plan in a pull request description
- If the user ask to open a pull request as feature then start the PR title with "feat:" prefix and also add one line about the feature into `CHANGELOG.md`
- Each changelog entry should follow the date of the PR in YYYY-MM-DD format. Example: Something was updated (2026-01-01).
- Before opening or updating a pull request, format the code
- When merging pull request, squash and merge commits and use the PR description as the commit message

### datetime

- Use naive UTC datetimes everywhere
- When using datetime class use `import datetime.datetime` and use `datetime.datetime` and `datetime.timedelta` as type hints
- Instead of `datetime.datetime.utcnow()` use `native_datetime_utc_now()` that is compatible across Python versions

### Enum

- For string enums, both members and values must in snake_case

### Pytest

- Never use test classes in pytest
- `pytest` tests should not have stdout output like `print`
- Instead of manual float fuzzy comparison like `assert abs(aave_total_pnl - 96.6087) < 0.01` use `pytest.approx()`
- Don't use logger.info() or logger.debug() inside test and fixture function bodies unless specifically asked

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
