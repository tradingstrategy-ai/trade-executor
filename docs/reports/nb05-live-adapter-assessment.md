# NB05 live adapter assessment

Assessment date: 29 October 2026. Source: [vault-notebooks PR #3](https://github.com/tradingstrategy-ai/vault-notebooks/pull/3),
revision `243217e7fa935baee18894048f84daa3d000626d`.

The main executor gaps are native Lighter pool execution, Atoma request and
claim accounting, early classification of delayed redemptions, GMX liquidity
execution and Symbiotic withdrawal support. Existing family adapters also need
deployment-specific validation before this universe can trade live.

This change is an **assessment and reproducible checks only**. It does not
implement adapters, convert the strategy, change dependencies, deploy a vault
or perform funded transactions. The existing local strategy export explicitly
rejects live universe creation and decisions; removing that guard alone would
not make the strategy executable.

## Population and evidence

[The machine-readable inventory](nb05-live-adapter-coverage.json) contains one
record for every one of the **347** source identities, including untraded
candidates and Fluid cash management. Identity is dataset chain plus address;
native dataset identifiers are retained separately from custody/signing chains.
The saved ledger reconciles to **54 traded vaults and 597 successful vault
trades**. Its 200 CCTP trades are not vault trades.

The recorded universe has 343 retained parent identities and four added tagged
Hypercore identities. Lighter LLP is excluded because this strategy does not
supply its staked-LIT prerequisite. The inventory does not re-screen inherited
membership against later lifetime returns.

| Metadata protocol | Sources | Factory-selected adapter / assessment |
| --- | ---: | --- |
| Morpho | 180 | 46 V1, 134 V2; complete immediate-redemption certification remains missing. |
| Lagoon Finance | 47 | Lagoon; validate request, actual settlement and claim per deployment. |
| Upshift | 27 | 17 multi-asset, 10 standard; preserve their different redemption flows. |
| Yearn | 21 | 11 Compounder, 10 V3; certify exact deployments and owner-scoped capacity. |
| Hyperliquid | 13 | Existing native route; validate funding, fees, lockups and withdrawals. |
| IPOR Fusion | 12 | Existing adapter; verify admission and redemption liquidity. |
| AUTOfinance / Tokemak | 8 | AutoPool; exact-vault guarded lifecycle evidence needed. |
| Fluid | 6 | Five Fluid and one Kiln; validate release of the parked-cash allocation. |
| Lighter | 6 | No executor public-pool route; existing cash-account tests cover another product. |
| Generic ERC-4626 label | 4 | All four select Yearn Compounder, not a generic adapter. |
| Euler | 4 | Two Euler and two Euler Earn; exact deployment validation needed. |
| 40acres | 3 | Existing adapter; historical Aerodrome redemption reverted. |
| Accountable | 3 | Synchronous deposits and operator-settled exits; selected deployments are on Monad. |
| Ember | 3 | Operator pays directly; no depositor claim transaction. |
| Gains Network | 3 | Synchronous deposits and epoch-gated redemption request/claim. |
| Atoma | 2 | Adapter merged upstream; executor receipt parsing and claim accounting need integration. |
| Plutus | 2 | Hedge is asynchronous; preserve the standard Dolomite variant separately. |
| D2 Finance | 1 | HYPE++; actual phases and account eligibility replace synthetic research windows. |
| GMX | 1 | GM liquidity shares; generic deposit manager explicitly unsupported. |
| Symbiotic | 1 | Reader exists; adapter deallocation/withdrawal lifecycle remains uncertified. |
| **Total** | **347** | **20 metadata protocol slugs** |

EVM adapter classes are resolved from the executor's pinned eth-defi factory
source, without executing constructors. Native readers are recorded separately
and are not returned by that factory. Feature precedence matters: Morpho flags win
over overlapping Yearn/Euler flags, and Kiln wins for one Fluid-labelled vault.
This is class selection, not on-chain implementation detection or validation.

The inventory joins the checked-in July corrective fork report and its HYPE++
and permission amendments in that order. **71** source identities have a
historical attempt; **276** do not appear in those reports. Each matching record
retains the report, status, operation, revisions and fork blocks. A displayed
simulated success can have a null raw `attempt.result`, so both fields are
retained. A closed-deposit Guard check proves policy validation only; injected
liquidity and mock settlement do not prove live exit availability.

Every record includes related executor tests and the remaining deployment
evidence requirement. Related tests are family references, not a claim that
they exercised that address. No source is newly certified by this assessment.
Recorded capability, permission and asset fields describe the 27 October input
snapshot; they are not fresh account-capacity observations.

## Gaps in the execution surfaces

### `perform-test-trade`

- Lighter public pools currently enter generic vault handling. Add a separate
  native pool route and resumable mint/burn/withdrawal lifecycle; the existing
  `lagoon-lighter-test-trade` exercises cash transfers, not pool investment.
- The public command has three `make_test_trade()` call sites and none passes
  `sync_state_callback`. State is written at command completion. Its internal
  cross-chain helper already exposes checkpoints, but the public command does
  not make them durable.
- Exercise deposit and exit as separate resumable actions when settlement spans
  invocations. Distinguish pending settlement, unavailable admission, unsupported
  simulation, closed-deposit Guard validation and a completed funded round trip.
- Atoma requires current-implementation forks. Upstream's lifecycle tests use
  Arbitrum block **519150162**; the canonical **487039644** predates that
  implementation. Historical decoder coverage and current execution are separate.

### `decide_trades()` and `TradeExecution`

- **23 sources** need earlier delayed-redemption classification: three Gains,
  three Ember and seventeen multi-asset Upshift. Live routing knows manager
  directionality, but allocation and CCTP planning act before routing stamps
  `vault_async_flow`. Pending proceeds must not fund same-cycle buys. Do not
  classify standard Upshift redemptions or these families' deposits as queued.
- Atoma request parsing currently rebuilds a redemption request after mining.
  The new manager rechecks unlocked shares and duplicate epoch requests even
  when `check_enough_token=False`. Escrow can therefore make parsing fail.
  Retain the preparation evidence and parse the receipt without resubmitting or
  rerunning admission against changed state.
- A valid zero-proceeds redemption fails `convert_vault_flow_analysis()` and
  later positive-proceeds trade assertions. Support evidenced total-loss vault
  exits through trade completion, PnL and reporting; keep other trade invariants.
- Atoma needs separate handling of escrowed shares, settled claim liabilities
  and realised cash. Settlement burns do not fund allocations. Refresh haircut
  valuations, leave illiquid claims unresolved, and use actual net receipt
  amounts without charging the exit fee twice.
- Integrate Atoma aggregate reconciliation and atomic supersession of component
  tickets. Recover already-mined claims; distinguish indexer lag and unsupported
  implementations from a request that can safely be retried.
- Three USDC-accounted Upshift vaults do not accept USDC deposits: **Upshift Clear
  RWA**, **Sentora USD Earn** and **LiquityV2 Airdrop Farming Vault**. They require
  accepted-token conversion, approvals and accounting before a USDC strategy can
  fund them. The existing incompatible-asset refusal is appropriate meanwhile.
- Test the actual decision-returned trades through execution and settlement,
  including restart, locks, fees, cash caps, direct operator payouts and NAV
  continuity. Native API operations must not acquire invented EVM receipts.

### `lagoon-deploy-vault`

- Lighter dataset chain **9998** is not normalised to Ethereum custody chain
  **1**. The ERC-4626 address collector attempts to checksum `lighter-pool-*`
  and raises `ValueError`; token collection must also exclude synthetic assets.
  Keep Lighter signing chain **304** separate.
- Current API-key activation requires Ethereum as the source chain. NB05's hub
  is Arbitrum, so consume upstream's Ethereum-satellite activation coordinator
  rather than treating the source-only ceremony as multichain pool support.
- Atoma needs the upstream `whitelistAtoma()` selection and compatible deployed
  bytecode. Existing broad ERC-4626 permissions are additive: remove them or
  replace the Guard/module before relying on the narrower policy.
- GMX liquidity requires a dedicated request/execution transaction adapter and
  matching deployment policy. Existing GMX perpetual exchange-account plumbing
  does not establish GM liquidity support.
- Validate every real custody chain, CCTP destination and exact contract against
  deployment configuration. Native dataset identifiers are not RPC chains.

## Upstream support and remaining evidence

[Lighter Core support, PR #1668](https://github.com/tradingstrategy-ai/web3-ethereum-defi/pull/1668),
is merged. Reuse `LighterPoolExecutor`, operation serialisation, live raw-share
sizing and secure withdrawal helpers. Persist original signed identity before
submission and reconcile uncertain operations before preparing another. Pool
selection is an API-key software policy; the Ethereum Guard restricts custody
calls and cannot enforce that choice. Account equity and pool holdings must not
be summed until their inclusion relationship is demonstrated. A funded sequencer
round trip and aggregate NAV conservation remain unverified upstream; Anvil
cannot process Lighter's sequencer lifecycle. See the
[execution hand-off](https://github.com/tradingstrategy-ai/web3-ethereum-defi/blob/7b2b43ef32bb0dd517dfe428976cfb2af10217dd/docs/README-lighter-core-execution.md).

[Atoma PR #1695](https://github.com/tradingstrategy-ai/web3-ethereum-defi/pull/1695)
merged on 29 October after the initial assessment. Reviewed head
`1fdb853a964f180aafd12789bae65a8968ad0e58` supplies the manager, persistent epoch
tickets, guarded request/claim, aggregate recovery and zero-payout analysis.
Its [execution notes](https://github.com/tradingstrategy-ai/web3-ethereum-defi/blob/1fdb853a964f180aafd12789bae65a8968ad0e58/docs/README-atoma-execution.md)
record fork and provider evidence and the remaining consumer responsibilities.
The executor's pinned dependency remains
`d19a5a6b9fcd29914b8749fcb438cf068002850d`; this assessment does not update it.
The snapshot's null Atoma capability must not be read as the merged adapter's
current capability.

The [GMX adapter](https://github.com/tradingstrategy-ai/web3-ethereum-defi/blob/752decd1ed56a1fff224ffa35492d494a02bf942/eth_defi/gmx/vault.py)
explicitly rejects generic transaction construction. The
[Symbiotic reader](https://github.com/tradingstrategy-ai/web3-ethereum-defi/blob/752decd1ed56a1fff224ffa35492d494a02bf942/eth_defi/erc_4626/vault_protocol/symbiotic/vault.py)
deliberately does not certify inherited synchronous execution. These two sources
were outside the earlier 54-traded-vault assessment. Yearn Compounder has
**15** factory-selected sources, including four under the generic protocol
label; certification must follow class selection rather than display labels.

## Reproduction and checks

The licensed metadata and saved state remain outside this repository. Only
sanitised coverage fields are retained. The JSON records their SHA-256 values,
the classifier revision/source hash, assessment/probed source hashes and
historical report hashes. It contains no private keys, RPC URLs or signed data.

Run from a prepared checkout, substituting the recorded input paths:

```shell
source .local-test.env && PYTHONPATH="$PWD:$PYTHONPATH" poetry run python \
  scripts/assess-nb05-vault-adapters.py \
  --notebook /path/to/source.ipynb \
  --metadata /path/to/vault-metadata.json \
  --state /path/to/notebook-backtest-state.json \
  --eth-defi-repo /path/to/web3-ethereum-defi
```

In a worktree with empty submodule directories, use the installed parent's
Poetry environment and put this checkout plus the parent dependency source
directories first on `PYTHONPATH`, as required by the repository instructions.
The classifier itself is read with `git show` at the executor's pinned revision,
independently of that dependency checkout's current revision. The diagnostic
probe imports use the installed environment; their relevant executor sources
are fingerprinted in the report. The command refuses a population other than
347 by default, missing/duplicate identities and overwriting an input. It never
executes notebook cells, constructs a vault or performs network requests.

```shell
source .local-test.env && PYTHONPATH="$PWD:$PYTHONPATH" poetry run pytest \
  tests/analysis/test_vault_adapter_coverage.py -q
```

The tests cover source selection, LLP/private exclusions, overlapping-feature
precedence, custody identities, multi-asset gaps, amended historical evidence,
trade deduplication and rejection of incomplete inputs. The committed inventory
also reconciles all 347 records, 54 traded vaults and 597 vault trades without
requiring the licensed datasets. The probes report observed defects rather than
tests that require those defects to remain unfixed.

Read-only probes reproduced the Lighter address-collection error, zero-proceeds
assertion, missing early feature classifications and absent public-command
checkpoint arguments. No new fork lifecycle, live provider test, financial
backtest, funded round trip or deployment was executed.
