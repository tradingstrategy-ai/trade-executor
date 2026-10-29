Here are some internal development guides for trade-executor package.

[See the full documentation on the website](https://tradingstrategy.ai/docs/index.html).

For the shared mechanism for data, transactions, journaling and checkpoints,
and the permission required before adding another system, see
[state persistence](./state-persistence.md).

For what a strategy module declares, which parameters the engine reads and
what does not belong in `Parameters`, see
[strategy parameters](./strategy-parameters.md).

For Lagoon guard redeployment and Safe proposal recovery, see the
[vault deployment guide](https://tradingstrategy.ai/docs/deployment/vault-deployment.html#upgrading-the-guard-smart-contract).

For guarded ERC-4626 test trades, see the
[Arbitrum manual tutorial](../scripts/lagoon/manual-erc4626-cli.sh) and its
[funded run report](reports/erc4626-manual-live-2026-10-29.md).

For the Ethereum Lighter account activation and public NAV wiring, see
[Lagoon Lighter deployment](./lagoon-lighter-deployment.md).

For operator setup and external-account accounting, see
[Lighter exchange account integration](../tradeexecutor/exchange_account/README-Lighter.md).

## Shared vault admission rule

Strategies and charts use `tradeexecutor.strategy.vault_closure`: `is_vault_closed()` reports closure, while `can_interact_with_vault()` applies interaction policy. See [protocol cases and API controls](vault-closure.md). Do not reinterpret raw maximum-capacity fields at call sites.
