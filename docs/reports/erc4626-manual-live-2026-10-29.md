# ERC-4626 manual live run — 2026-10-29

The actual `scripts/lagoon/manual-erc4626-cli.sh` completed on Arbitrum mainnet,
17:33–17:52 UTC, including two inspected resumptions. It deployed Lagoon, subscribed
10 USDC, opened and closed five 1 USDC positions through separate
`perform-test-trade` invocations, checked accounts after each exit, settled the
terminal NAV and redeemed the investor. No protocol liquidity was injected.

The run used the local executor changes and local shared eth-defi sizing fixes
described in the [subplan](../plans/nb05-generic-erc4626-test-trade-subplan.md).
These changes were uncommitted at execution time. The executor integration pins
the shared sizing fix published in [eth-defi PR #1700](https://github.com/tradingstrategy-ai/web3-ethereum-defi/pull/1700).
This is not evidence for the original dependency pin alone or all 347 source vaults.

## Deployment and results

- Investor/deployer: `0xdcc6D3A3C006bb4a10B448b1Ee750966395622c6`.
- [Lagoon vault](https://arbiscan.io/address/0x97C98da1c70B00A41E96fA93DFF9F34967984540): `0x97C98da1c70B00A41E96fA93DFF9F34967984540`.
- Safe: `0xF6197814EAE8FE65CaDb2A94893243Ab678Af045`.
- Guard/module: `0x3D1A0E39De84886aD5ED344b6Fb03B784Bb829e1`.
- Deployment start block: `519425186`.

| Implementation | Deposited USDC | Returned USDC | Deposit | Redemption |
| --- | ---: | ---: | --- | --- |
| Generic ERC-4626 (Euler) | 1 | 0.999999 | [Transaction](https://arbiscan.io/tx/0x81624d377e0e170e8087ff90a341c548cea8c90eacd093192f4b3aa288f13e0e) | [Transaction](https://arbiscan.io/tx/0x58e5f5e3702c0d9099546f13fe3fc27fc0bf6f64cb36e023fbbd52dc78dc8631) |
| Morpho V1 | 1 | 1 | [Transaction](https://arbiscan.io/tx/0x0d7c41bc0a0ef557c014ac596002d86cc2a3fb2184862aafbb77b4e60f573488) | [Transaction](https://arbiscan.io/tx/0x8912b8b7b8d056e83f52a185aeb83d96031092ed7dbea9fe00f6fb4c0027d79f) |
| Morpho V2 | 1 | 1 | [Transaction](https://arbiscan.io/tx/0x1839303c77c2346b28f916c31d5ef5ada3669f93500ff4158ddf99a071e8a35a) | [Transaction](https://arbiscan.io/tx/0x9f7e6099a2c9554d2761cd6cc120d68b0409eb1009cb1c99f0846092b26bcda3) |
| Yearn V3 | 1 | 0.999999 | [Transaction](https://arbiscan.io/tx/0x17c40da5fb9983474d40449e6cdb307706f59a1f43dd25a6a4b592b56b9e20ed) | [Transaction](https://arbiscan.io/tx/0xf9fe3045ad75421769191c7400a8e2a253ead378bfc3249971542c70b2296bac) |
| Yearn Compounder | 1 | 0.999999 | [Transaction](https://arbiscan.io/tx/0xb947d59a08712bda51892cef315100cd2699fa49d16768150c575008d3dbbe8b) | [Transaction](https://arbiscan.io/tx/0x10c73964387c9aa38411f290146846dac0f9e609a2ce872c89f8e1db66493e27) |

All ten trades succeeded. The ledger has five closed positions, no open/frozen
positions and zero reserve cash after investor redemption. The final audit read
zero protocol shares in the Safe and zero Lagoon shares in the investor wallet.
Lagoon's Safe USDC balance is zero.

The [investor claim](https://arbiscan.io/tx/0x58e863207a59b51b3e72ec3c65d8f7e00ab04bec29caa34aeb9d74d124aed2ca)
returned **9.999997 USDC**. The total protocol rounding loss was **0.000003 USDC**.
Investor USDC changed from 25.175780 to 25.175777; ETH changed from
0.006449711482172622 to 0.006093945999386622. The run spent
**0.000355765482786 ETH** across deployment, configuration, funding, trading and
redemption (wallet nonces 1578–1618). The earlier USDC-to-ETH swap is excluded.

## Issues found and handling

1. **`check-wallet` omitted strategy parameters.** The first run stopped after
   deployment and initialisation, before subscription. Pass `mod.parameters` to
   the existing universe wrapper and correct the Anvil override's default to
   `False`. Resumed the same deployment with `--subscribe`; no second deployment.
   The fork tutorial now includes `check-wallet` in setup. Its full round trip
   and both existing mocked wallet regressions passed after the fix.
2. **HyperSync was behind the newest Arbitrum blocks.** After the first deposit,
   the close invocation refused to proceed because it could not prove that the
   unconfirmed range had no external settlement. Unsetting `HYPERSYNC_API_KEY`
   selected the existing RPC log reader. Resuming without deployment/subscription
   flags closed the saved position; the buy was not repeated. No state edits,
   reconciliation bypass or additional persistence system were used.
3. **Existing wrapped-call diagnostics retry valid empty return data.** Each
   successful close returned empty data from the module's `eth_call`; the RPC
   fallback treated that as missing historical state and retried. Routing records
   the diagnostic error and then executes normally. This added roughly a minute
   per close but did not prevent execution. The diagnostic/provider compatibility
   issue remains a follow-up; it was not changed during this run.

The tutorial now preserves `MIN_GAS_BALANCE` in its public resume configuration,
defaulting to 0.001 ETH, matching the existing Lagoon funding/redemption commands.
It also checks its `check-wallet` API-key prerequisite before deployment and
documents the RPC-reader workaround.

## Reproducibility and artefacts

The run used the parent Poetry environment with worktree and local eth-defi source
precedence, `UNIT_TESTING=false`, `SIMULATE=false`, `MIN_GAS_BALANCE=0.001`,
`SUBSCRIPTION_USDC=10` and `TEST_AMOUNT=1`. Credentials came from the existing test
environment and were not written into the public resume configuration.

The private local run directory is
`~/.tradingstrategy/examples/erc4626-cli-20261029-live/`. It contains
`run.log`, the frozen `strategy.py`, public `env.sh`, the deployment artefact and
the authoritative executor state at `state/erc4626-cli.json`. Do not deploy again
into this directory or resubscribe it to reproduce this completed test.

Subsequent PR cleanup moved the same target list from `Parameters` to a module
constant and tightened the public close-only guard before treasury sync. The
frozen strategy in the run directory records the version actually executed.

This proves the five listed deployments accepted these small live round trips
at this time. It does not establish large-order capacity, future liquidity,
every deployment of these protocols, or recovery from an interrupted broadcast.
