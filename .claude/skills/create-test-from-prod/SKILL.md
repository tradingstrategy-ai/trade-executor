---
name: create-test-from-prod
description: Download state from a running trade executor and scaffold an acceptance test case
---

# Create test from production

Download the state file from a running trade-executor instance and scaffold a pytest acceptance test module.

## Inputs

- **Trade executor web server address** — URL of a running trade-executor (e.g. `https://executor.example.com`)
- **Test case name** — identifier for the test (e.g. `my_new_test`); will be normalised to snake_case and any leading `test_` prefix is stripped for the directory name

## Process

### 1. Fetch executor metadata and source

Run these two curl commands to gather information about the running executor:

```shell
curl -s <url>/metadata | python -m json.tool
```

```shell
curl -s <url>/source
```

From the `/metadata` JSON response, extract:

- **`name`** — strategy name (used in the test module docstring)
- **`on_chain_data.chain_id`** — blockchain chain ID; map to the appropriate environment variable name:
  - `1` → `JSON_RPC_ETHEREUM`
  - `137` → `JSON_RPC_POLYGON`
  - `8453` → `JSON_RPC_BASE`
  - `42161` → `JSON_RPC_ARBITRUM`
  - `56` → `JSON_RPC_BNB`
  - `43114` → `JSON_RPC_AVALANCHE`
- **`on_chain_data.asset_management_mode`** — one of `enzyme`, `hot_wallet`, `lagoon`, etc.
- **Vault addresses** from `on_chain_data` (include whichever are present):
  - `vault` → `VAULT_ADDRESS`
  - `generic_adapter` → `VAULT_ADAPTER_ADDRESS`
  - `payment_forwarder` → `VAULT_PAYMENT_FORWARDER_ADDRESS`
  - `deployment_block` → `VAULT_DEPLOYMENT_BLOCK_NUMBER`

### 2. Identify matching local strategy module

Compare the source code from `/source` against local strategy files:

1. Read files in `strategies/` and `strategies/test_only/`
2. Find a file whose contents match the downloaded source (exact or near-exact match)
3. If no match is found, present the list of strategy files to the user with `AskUserQuestion` and ask them to select the correct one or provide a path

Record whether the strategy is in `strategies/` or `strategies/test_only/` and its filename.

### 3. Create test directory and download state file

```shell
mkdir -p tests/mainnet_fork/<test_case_name>
curl -s <url>/state -o tests/mainnet_fork/<test_case_name>/state.json
```

Verify the download succeeded and the file is non-empty.

### 4. Generate test module skeleton

Create `tests/mainnet_fork/<test_case_name>/test_<test_case_name>.py` using the template below.

Fill all `<PLACEHOLDER>` values from the metadata and strategy identification steps. Only include vault-specific environment variables that are actually present in the metadata.

```python
"""<Strategy name> acceptance test.

- Does <Chain name> mainnet fork
- Archive node needed
- State file downloaded from <url>
"""
import os.path
import secrets
from pathlib import Path
from unittest import mock

import pytest
from _pytest.fixtures import FixtureRequest

from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from tradeexecutor.cli.commands.app import app
from tradeexecutor.utils.hex import hexbytes_to_hex_str


pytestmark = pytest.mark.skipif(
    not os.environ.get("<JSON_RPC_CHAIN>") or not os.environ.get("TRADING_STRATEGY_API_KEY"),
    reason="Set <JSON_RPC_CHAIN> and TRADING_STRATEGY_API_KEY environment variables to run this test",
)


@pytest.fixture()
def anvil(request: FixtureRequest) -> AnvilLaunch:
    """Do <Chain name> mainnet fork."""
    mainnet_rpc = os.environ["<JSON_RPC_CHAIN>"]
    anvil = launch_anvil(mainnet_rpc)
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def state_file() -> Path:
    """State file downloaded from production."""
    p = Path(os.path.join(os.path.dirname(__file__), "state.json"))
    assert p.exists(), f"{p} missing"
    return p


@pytest.fixture()
def strategy_file() -> Path:
    """The strategy module for this executor."""
    p = Path(os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "strategies", "<strategies_or_test_only>", "<strategy_filename>.py",
    ))
    assert p.exists(), f"{p.resolve()} missing"
    return p


@pytest.fixture()
def environment(
    anvil: AnvilLaunch,
    state_file: Path,
    strategy_file: Path,
    persistent_test_client,
) -> dict:
    """Passed to init and start commands as environment variables."""
    environment = {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hexbytes_to_hex_str(secrets.token_bytes(32)),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "<asset_management_mode>",
        "UNIT_TESTING": "true",
        "UNIT_TEST_FORCE_ANVIL": "true",
        "LOG_LEVEL": "disabled",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "SKIP_SAVE": "true",
        "CACHE_PATH": str(persistent_test_client.transport.cache_path),
    }
    return environment


def test_<test_case_name>(environment: dict):
    """<Description>."""
    pass
```

### 5. Add vault-specific environment variables

If the metadata contains vault addresses, add them to the `environment` fixture dict. Common patterns:

**Enzyme vaults** (`asset_management_mode: enzyme`):
```python
"VAULT_ADDRESS": "<vault>",
"VAULT_ADAPTER_ADDRESS": "<generic_adapter>",
"VAULT_PAYMENT_FORWARDER_ADDRESS": "<payment_forwarder>",
"VAULT_DEPLOYMENT_BLOCK_NUMBER": "<deployment_block>",
```

**Lagoon vaults** (`asset_management_mode: lagoon`):
```python
"VAULT_ADDRESS": "<vault>",
"VAULT_ADAPTER_ADDRESS": "<generic_adapter>",
```

For other asset management modes, include whatever vault-related fields are present in `on_chain_data`.

## Reference files

- Test pattern examples: `tests/mainnet_fork/test_correct_interest_not_accrued.py`, `tests/mainnet_fork/test_correct_accouting_closed_position_has_ausdc.py`
- API endpoints: `tradeexecutor/webhook/api.py` (routes: `/state`, `/metadata`, `/source`)
- Root conftest with `persistent_test_client` fixture: `tests/conftest.py`
- Strategy files: `strategies/` and `strategies/test_only/`

## Notes

- The `persistent_test_client` fixture is defined in `tests/conftest.py` and is available to all tests automatically
- The test function body is intentionally left as `pass` — the user will fill in the actual test logic
- If the strategy file is in `strategies/` (not `test_only/`), adjust the relative path in the `strategy_file` fixture accordingly (remove the `test_only` segment)
- State files can be large (10+ MB); the download may take a moment
