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

9. Clean up import clutter in the `check-wallet` CLI command

   Status:
   - Done

   Scope:
   - Remove duplicate imports and obvious unused imports in `tradeexecutor/cli/commands/check_wallet.py`
   - Keep behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/cli/test_check_wallet_hyper_ai.py --log-cli-level=info`

10. Clean up import clutter in the `export` CLI command

   Status:
   - Done

   Scope:
   - Remove duplicate imports and obvious unused imports in `tradeexecutor/cli/commands/export.py`
   - Keep behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/cli/test_cli_commands.py -k test_cli_export --log-cli-level=info`

11. Clean up duplicate imports in state storage

   Status:
   - Done

   Scope:
   - Remove duplicate imports and obvious unused imports in `tradeexecutor/state/store.py`
   - Keep serialisation behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/test_store.py --log-cli-level=info`

12. Clean up duplicate imports in reserve state handling

   Status:
   - Done

   Scope:
   - Remove duplicate imports and obvious unused imports in `tradeexecutor/state/reserve.py`
   - Keep reserve accounting behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/test_state.py -k test_update_reserves --log-cli-level=info`

13. Clean up import clutter in the `check-accounts` CLI command

   Status:
   - Done

   Scope:
   - Remove duplicate imports and obvious unused imports in `tradeexecutor/cli/commands/check_accounts.py`
   - Keep command behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/enzyme/test_enzyme_end_to_end.py::test_enzyme_check_accounts --log-cli-level=info`

14. Clean up import clutter in the `check-position-triggers` CLI command

   Status:
   - Done

   Scope:
   - Remove duplicate imports and obvious unused imports in `tradeexecutor/cli/commands/check_position_triggers.py`
   - Keep command behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/mainnet_fork/test_enzyme_credit_position.py::test_enzyme_credit_position_redemption_check_triggers --log-cli-level=info`

15. Clean up import clutter in the `console` CLI command

   Status:
   - Done

   Scope:
   - Remove duplicate imports and obvious unused imports in `tradeexecutor/cli/commands/console.py`
   - Keep console command behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/cli/test_cli_commands.py -k test_cli_console --log-cli-level=info`

16. Clean up import clutter in the `distribute-gas-funds` CLI command

   Status:
   - Done

   Scope:
   - Remove obvious unused imports in `tradeexecutor/cli/commands/distribute_gas_funds.py`
   - Keep dry-run and chain selection behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/cli/test_cli_distribute_gas_funds.py --log-cli-level=info`

17. Clean up import clutter in the `trading-pair` CLI command

   Status:
   - Done

   Scope:
   - Remove duplicate imports and obvious unused imports in `tradeexecutor/cli/commands/trading_pair.py`
   - Keep command output behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/cli/test_cli_trading_pair.py --log-cli-level=info`

18. Clean up duplicate imports in the timer utility

   Status:
   - Done

   Scope:
   - Remove duplicate imports and obvious unused imports in `tradeexecutor/utils/timer.py`
   - Keep timed task logging behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/test_decision_trigger.py --log-cli-level=info`

19. Clean up duplicate imports in trade retry helpers

   Status:
   - Done

   Scope:
   - Remove duplicate imports and obvious unused imports in `tradeexecutor/state/retry.py`
   - Keep retry and rebroadcast behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/ethereum/test_repair_rebroadcast.py --log-cli-level=info`

20. Clean up duplicate imports in portfolio state handling

   Status:
   - Done

   Scope:
   - Remove duplicate imports and obvious unused imports in `tradeexecutor/state/portfolio.py`
   - Keep reserve allocation and portfolio accounting behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/test_state.py -k test_not_enough_cash --log-cli-level=info`

21. Clean up duplicate imports in frozen-position handling

   Status:
   - Done

   Scope:
   - Remove duplicate imports in `tradeexecutor/state/freeze.py`
   - Keep failed-trade freezing behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/test_unclean_state.py --log-cli-level=info`

22. Clean up duplicate imports in interest state handling

   Status:
   - Done

   Scope:
   - Remove duplicate imports in `tradeexecutor/state/interest.py`
   - Keep interest tracking data behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/interest/test_state_short.py -k test_open_short --log-cli-level=info`

## Execution notes

- Issues should be landed one by one, with the relevant verification run after each change.
- Where issues are independent, they can be implemented in parallel and then integrated back into this branch.
- No formatting-only changes.
