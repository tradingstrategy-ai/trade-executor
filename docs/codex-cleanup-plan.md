# Codex cleanup plan

This document tracks small, low-risk cleanups that can be completed and verified independently.

## Issues

1. Fix packaging and installation metadata drift

   Status:
   - Done

   Scope:
   - Fix `poetry check` failure caused by the undeclared `setuptools` extra reference in `pyproject.toml`
   - Align installation guidance in `README.md` with the current Python version and current extras
   - Keep the change limited to packaging metadata and contributor-facing docs

   Verification:
   - `poetry check`

2. Clean up webhook dead code and make the notify endpoint explicit

   Status:
   - Done

   Scope:
   - Remove dead helper code in `tradeexecutor/webhook/api.py`
   - Replace the placeholder `/notify` success response with an explicit “not implemented” API response
   - Add or update webhook coverage for the endpoint behaviour

   Verification:
   - `source .local-test.env && poetry run pytest tests/web/test_webhook_api.py --log-cli-level=info`

3. Remove import and startup-command clutter in active CLI modules

   Status:
   - Done

   Scope:
   - Remove duplicate and unused imports in `tradeexecutor/cli/commands/start.py`
   - Remove duplicate imports in `tradeexecutor/strategy/runner.py`
   - Keep behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/cli/test_cli_since_last_cycle_end.py --log-cli-level=info`

4. Clean up duplicate imports in the legacy Uniswap v2 execution path

   Status:
   - Done

   Scope:
   - Remove duplicate imports and similar no-behaviour-change clutter in `tradeexecutor/ethereum/uniswap_v2/uniswap_v2_execution_v0.py`
   - Keep the legacy code path working for strategies and tests that still import it

   Verification:
   - `source .local-test.env && poetry run pytest tests/test_approval_in_terminal.py --log-cli-level=info`

5. Clean up import clutter in the `check-universe` CLI command

   Status:
   - Done

   Scope:
   - Remove duplicate imports and obvious unused imports in `tradeexecutor/cli/commands/check_universe.py`
   - Keep behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/cli/test_cli_check_universe.py --log-cli-level=info`

6. Clean up import clutter in the `blacklist` CLI command

   Status:
   - Done

   Scope:
   - Remove duplicate imports and obvious unused imports in `tradeexecutor/cli/commands/blacklist.py`
   - Keep behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/cli/test_cli_blacklist.py --log-cli-level=info`

7. Clean up small logging-module clutter in webhook HTTP logging

   Status:
   - Done

   Scope:
   - Remove obvious unused imports in `tradeexecutor/webhook/http_log.py`
   - Keep HTTP log behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/cli/test_http_log.py --log-cli-level=info`

8. Clean up duplicate imports in strategy metadata

   Status:
   - Done

   Scope:
   - Remove duplicate imports in `tradeexecutor/state/metadata.py`
   - Keep metadata serialisation and webhook-facing behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/web/test_webhook_api.py --log-cli-level=info`

## Execution notes

- Issues should be landed one by one, with the relevant verification run after each change.
- Where issues are independent, they can be implemented in parallel and then integrated back into this branch.
- No formatting-only changes.
