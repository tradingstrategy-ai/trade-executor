# Codex cleanup plan

This document tracks small, low-risk cleanups that can be completed and verified independently.

## Issues

Completed in this branch:
- Issues `1` to `22`

Pending:

23. Clean up duplicate compat imports in CLI bootstrap helpers

   Status:
   - Pending

   Scope:
   - Remove duplicate `native_datetime_utc_now` imports and similar no-behaviour-change clutter in `tradeexecutor/cli/bootstrap.py`
   - Keep bootstrap and metadata behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/enzyme/test_enzyme_webhook_api.py::test_enzyme_metadata --log-cli-level=info`

24. Clean up duplicate compat imports in the close-position helper module

   Status:
   - Pending

   Scope:
   - Remove duplicate `native_datetime_utc_now` imports and similar low-risk clutter in `tradeexecutor/cli/close_position.py`
   - Keep close-all and close-position helper behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/cli/test_close_position.py::test_close_all --log-cli-level=info`

25. Clean up duplicate compat imports in the `close-all` CLI command

   Status:
   - Pending

   Scope:
   - Remove duplicate `native_datetime_utc_now` imports and obvious unused imports in `tradeexecutor/cli/commands/close_all.py`
   - Keep command behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/cli/test_close_position.py::test_close_all --log-cli-level=info`

26. Clean up duplicate compat imports in the `close-position` CLI command

   Status:
   - Pending

   Scope:
   - Remove duplicate `native_datetime_utc_now` imports and obvious unused imports in `tradeexecutor/cli/commands/close_position.py`
   - Keep command behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/cli/test_close_position.py::test_close_position_single --log-cli-level=info`

27. Clean up duplicate compat imports in the `correct-accounts` CLI command

   Status:
   - Pending

   Scope:
   - Remove duplicate compat imports and other obvious unused imports in `tradeexecutor/cli/commands/correct_accounts.py`
   - Keep command behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/enzyme/test_enzyme_end_to_end.py::test_enzyme_correct_accounts --log-cli-level=info`

28. Clean up duplicate compat imports in the `perform-test-trade` CLI command

   Status:
   - Pending

   Scope:
   - Remove duplicate `native_datetime_utc_now` imports and other obvious unused imports in `tradeexecutor/cli/commands/perform_test_trade.py`
   - Keep command behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/enzyme/test_enzyme_end_to_end.py::test_enzyme_perform_test_trade --log-cli-level=info`

29. Clean up duplicate compat imports in the `repair` CLI command

   Status:
   - Pending

   Scope:
   - Remove duplicate `native_datetime_utc_now` imports and obvious unused imports in `tradeexecutor/cli/commands/repair.py`
   - Keep repair command behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/mainnet_fork/test_repair_frozen_credit_position.py::test_repair_frozen_credit_position --log-cli-level=info`

30. Clean up duplicate compat imports in the `retry` CLI command

   Status:
   - Pending

   Scope:
   - Remove duplicate `native_datetime_utc_now` imports and obvious unused imports in `tradeexecutor/cli/commands/retry.py`
   - Keep retry command behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/ethereum/test_repair_rebroadcast.py::test_repair --log-cli-level=info`

31. Clean up duplicate compat imports in the `webapi` CLI command

   Status:
   - Pending

   Scope:
   - Remove duplicate `native_datetime_utc_now` imports and similar low-risk clutter in `tradeexecutor/cli/commands/webapi.py`
   - Keep web API and webhook behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/web/test_webhook_api.py --log-cli-level=info`

32. Clean up duplicate compat imports in vault settlement helpers

   Status:
   - Pending

   Scope:
   - Remove duplicate `native_datetime_utc_now` imports and similar low-risk clutter in `tradeexecutor/cli/settle_vault.py`
   - Keep vault settlement helper behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/lagoon/test_lagoon_e2e.py::test_cli_lagoon_settle --log-cli-level=info`

33. Clean up duplicate compat imports in test-trade helpers

   Status:
   - Pending

   Scope:
   - Remove duplicate `native_datetime_utc_now` imports and similar low-risk clutter in `tradeexecutor/cli/testtrade.py`
   - Keep test-trade helper behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/enzyme/test_enzyme_end_to_end.py::test_enzyme_perform_test_trade --log-cli-level=info`

34. Clean up duplicate compat imports in trading position state

   Status:
   - Pending

   Scope:
   - Remove duplicate `native_datetime_utc_now` imports and other obvious low-risk import clutter in `tradeexecutor/state/position.py`
   - Keep position accounting and profit calculation behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/test_state.py -k test_unrealised_profit_calculation --log-cli-level=info`

35. Clean up duplicate compat imports in state repair helpers

   Status:
   - Pending

   Scope:
   - Remove duplicate `native_datetime_utc_now` imports and similar no-behaviour-change clutter in `tradeexecutor/state/repair.py`
   - Keep repair behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/legacy/test_legacy_repair_trade.py::test_repair_trades --log-cli-level=info`

36. Clean up duplicate compat imports in uptime state handling

   Status:
   - Pending

   Scope:
   - Remove duplicate `native_datetime_utc_now` imports in `tradeexecutor/state/uptime.py`
   - Keep cycle-completion recording behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/test_live_cycle_scheduler.py --log-cli-level=info`

37. Clean up duplicate compat imports in state visualisation helpers

   Status:
   - Pending

   Scope:
   - Remove duplicate `native_datetime_utc_fromtimestamp` imports in `tradeexecutor/state/visualisation.py`
   - Keep visualisation serialisation and plotting behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/test_strategy_state_visualisation.py::test_visualise_strategy_state --log-cli-level=info`

38. Clean up duplicate compat imports in the live execution loop

   Status:
   - Pending

   Scope:
   - Remove duplicate compat imports in `tradeexecutor/cli/loop.py`
   - Keep live cycle scheduling behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/test_live_cycle_scheduler.py --log-cli-level=info`

39. Clean up obvious unused imports in the `lagoon-settle` CLI command

   Status:
   - Pending

   Scope:
   - Remove obvious unused imports in `tradeexecutor/cli/commands/lagoon_settle.py`
   - Keep lagoon settlement command behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/lagoon/test_lagoon_e2e.py::test_cli_lagoon_settle --log-cli-level=info`

40. Clean up obvious unused imports in the `enzyme-asset-list` CLI command

   Status:
   - Pending

   Scope:
   - Remove obvious unused imports in `tradeexecutor/cli/commands/enzyme_asset_list.py`
   - Keep command behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/cli/test_cli_commands.py -k test_cli_console --log-cli-level=info`

41. Modernise legacy `Union` usage in state type aliases

   Status:
   - Pending

   Scope:
   - Replace legacy `Union[...]` typing with native `|` syntax in `tradeexecutor/state/types.py`
   - Keep runtime behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/test_state.py -k test_serialize_panda_timestamp --log-cli-level=info`

42. Modernise legacy `Union` usage in timestamp utilities

   Status:
   - Pending

   Scope:
   - Replace legacy `Union[...]` typing with native `|` syntax in `tradeexecutor/utils/timestamp.py`
   - Keep timestamp conversion behaviour unchanged

   Verification:
   - `source .local-test.env && poetry run pytest tests/test_state.py -k test_serialise_timedelta --log-cli-level=info`

## Execution notes

- Issues should be landed one by one, with the relevant verification run after each change.
- Where issues are independent, they can be implemented in parallel and then integrated back into this branch.
- No formatting-only changes.
