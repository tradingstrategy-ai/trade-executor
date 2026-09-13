# Lighter exchange account integration

This document describes Lighter support in trade-executor for a Lagoon Safe on
Ethereum. It covers Safe-owned Lighter account activation, public account-value
tracking and Lagoon NAV. The executor does not submit Lighter orders.

## Overview

A Lighter account is represented as one external exchange-account position in
the strategy state. Its value is Lighter's public total equity, including
collateral and unrealised PnL. For a Lagoon vault the posted NAV is:

```text
Safe native-USDC balance + Lighter total equity + pending Lagoon settlement value
```

The Lighter API private key is needed only by an external execution tool. It is
not part of the strategy, executor environment, state or public deployment
reports.

## Deploy a Safe-owned account

`lagoon-deploy-vault` can create a fresh Safe-owned account and register a
Lighter API key during an Ethereum deployment:

```shell
export JSON_RPC_ETHEREUM="https://..."
export PRIVATE_KEY="0x..."
export VAULT_RECORD_FILE="/secure/path/lighter-vault-record.txt"
export GENERATE_LIGHTER_API_KEY=true
export LIGHTER_API_KEY_INDEX=4

trade-executor lagoon-deploy-vault
```

This is live Ethereum work: it spends gas and the accounted Lighter activation
deposit. It cannot run with `SIMULATE=true` because an Anvil fork cannot make a
new account visible to Lighter's sequencer.

The text record, Markdown report and executor deployment artefact contain only
the public account index, API-key slot, public key and transaction hashes. The
paired `vault-record.json` is the only file containing the generated Lighter
private key and is written with mode `0600`. Keep it in a secret manager; do
not put it in strategy configuration, logs, tickets or chat.

## Strategy setup

Use the public Lighter account index from the deployment report in the strategy
universe:

```python
from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pair

lighter_pair = create_lighter_exchange_account_pair(
    quote=usdc,
    account_index=123,
)
```

The Lighter pair must be the strategy's only external exchange-account pair.
The initial integration supports native Ethereum USDC and one Lighter account
per strategy. It is a valuation/accounting integration, not a trade-routing or
order-execution integration.

See `strategies/test_only/minimal_lighter_strategy.py` for a complete static
universe example.

## Account and NAV synchronisation

Initialise the strategy state and create the external-account position:

```shell
trade-executor init
trade-executor correct-accounts
```

`correct-accounts` and `lagoon-settle` work for this static universe without a
Trading Strategy API key. They read Lighter's unauthenticated public account
endpoint using the pair's account index.

During normal operation, `lagoon-settle` revalues the external account and
posts the Safe-plus-Lighter NAV. Run it only after a Safe-to-Lighter deposit or
Lighter-to-Safe withdrawal is visible in the public account response. The
public endpoint is current-state data, so it is not pinned to the Ethereum
block used for the Safe balance.

## Safety behaviour

- A negative, non-finite or malformed Lighter equity response aborts valuation
  and account correction. It is never written as a balance update or posted as
  a Lagoon NAV.
- A Lighter API failure fails closed; the executor does not reuse a stale
  external-account value for a new NAV.
- The Lighter API key is never required for valuation. The public account index
  is the only Lighter credential-like field allowed in strategy metadata.
- Pause the executor while collateral is in transit. Resume only after the
  public account value and Safe balance reflect the completed movement.

## Manual lifecycle check

`scripts/lagoon/manual-trade-executor-lighter.py` is a live, guarded manual
test. It deploys a temporary Lagoon vault, deposits accounted collateral,
syncs NAV with Typer commands, opens and closes a small ETH perpetual through
the separately installed Lighter SDK, withdraws to the Safe, and redeems the
shares. It requires `LIGHTER_TEST_PRIVATE_KEY` and `JSON_RPC_ETHEREUM` and
must not be used with `SIMULATE=true`.

## See also

- [Lagoon Lighter deployment guide](../../docs/lagoon-lighter-deployment.md)
- [Derive exchange account integration](./README-Derive.md)
- [eth-defi Lighter guard architecture](../../deps/web3-ethereum-defi/eth_defi/lighter/README-lighter-guard.md)
