# Lighter multi-subaccount support: gap analysis

## Goal

Let one Lagoon vault custody capital across several Lighter accounts:

- The Safe owns one Lighter **master account** and N **sub-accounts**.
- Each sub-account has its own API key, handed to an external Nautilus
  trading process that runs one sub-strategy per sub-account.
- `trade-executor` holds no trading logic for Lighter, but it must be able to
  deposit into and withdraw from any of the accounts, and rebalance trade
  capital between sub-accounts from `decide_trades()`.

This document records what the protocol allows, what each component does
today, what has to change, and how hard each change is. It is an analysis,
not an implementation plan. Nothing in this document has been built yet.

The analysis covers:

1. Lighter protocol facts that constrain the design
2. The guard smart contract (`GuardV0Base.sol` and `LighterLib.sol`)
3. `web3-ethereum-defi` deployment scripts and support functions
4. `trade-executor` deployment, routing and the `decide_trades()` loop
5. A Lighter EVM deployment

Sources were checked on 2026-10-26: the verified `ZkLighter` implementation
source (`0x831ef69bab8af8b1037a4961b8d0674b124e7008`, fetched from Sourcify),
the Lighter developer docs, the pinned `lighter-sdk` fork installed in the
project virtualenv, and the current `master` of both repositories.

## Lighter protocol facts that drive the design

| Fact | Evidence | Consequence |
|------|----------|-------------|
| One L1 address owns one master account plus sub-accounts. Master indices count up from 0; sub-account and pool indices count down from `2^48 - 2`. | Core concepts doc; `ZkLighter` comment that `lastAccountIndex` excludes sub-accounts | A sub-account index is not close to the master index. It must be looked up from `accountsByL1Address`. |
| Sub-account limit depends on the account tier: Standard 4, Plus 16, Premium 64. Sub-accounts share the tier of the master L1 address. | Account types doc | The number of Nautilus strategies per vault is bounded. Changing tier is an authenticated API call with a 24 h cooldown for downgrades. |
| A sub-account is created with the L2 transaction `CreateSubAccount` (`tx_type = 9`), signed by the **master account's API key**. No L1 call exists. | Signer SDK `sign_create_sub_account(...)`; tx type constants | The executor or a deployment script must hold a master API key to create sub-accounts. The Safe alone cannot do it. |
| Every account, master or sub-account, has its own API key slots 0-254. Slots 0-3 are reserved for the UI. Registering a key **replaces** whatever key was at that slot. | API keys doc | Nautilus and the executor must never share one slot. Each account needs one slot per signer. |
| A smart-contract owner registers a key with the L1 call `changePubKey(accountIndex, apiKeyIndex, pubKey)`. The contract validates only that `msg.sender` is a registered master and forwards `(accountIndex, masterAccountIndex)` as a priority request; the sequencer checks that `accountIndex` is the master or one of its sub-accounts. | `AdditionalZkLighter.changePubKey()` at lines 164-205 of the verified source; docs query endpoint answer "Yes" | The Safe can register keys on its sub-accounts with the same Safe-owner transaction it uses today. Nothing new on L1 is needed for key registration. |
| An L1 `deposit(_to, assetIndex, routeType, amount)` credits the **master** account of `_to`. There is no account-index parameter. | `AdditionalZkLighter.deposit()`; L1-and-Core token transfer doc: "This deposit path targets L1-controlled masters, not subaccounts" | Funding a sub-account is two steps: L1 deposit to master, then an L2 transfer master to sub-account. |
| An L2 `Transfer` (`tx_type = 12`) between accounts of the **same** owner needs only the source account's API key. No L1 signature. A fee from `transferFeeInfo` is charged on top of the amount and must match exactly. Minimum 1 USDC. | Deposits, transfers and withdrawals doc; SDK `transfer_same_master_account(...)` | Master to sub-account moves need the master key. Sub-account to master or sub-account to sub-account moves need the **source sub-account's** key. |
| A secure withdrawal (`tx_type = 13`) from any account goes only to the owner's L1 address and needs only that account's API key. Lighter normally claims it on L1, otherwise the Safe claims it with `withdrawPendingBalance(owner, ...)`. Delay is dynamic, about 20 minutes observed. | Deposits, transfers and withdrawals doc; current executor flow | A sub-account withdrawal to the Safe needs the sub-account key, or the alternative below. |
| The L1 call `withdraw(accountIndex, assetIndex, routeType, amount)` takes any account index and binds `masterAccountIndex` to `msg.sender`. The pending balance is credited to the master's L1 address when the batch is verified. | `AdditionalZkLighter.withdraw()` lines 502-525; `ZkLighter.sol` line 573 credits `_tx.masterAccountIndex` | The Safe can pull USDC out of any of its sub-accounts **without any API key**, through the guard's already whitelisted `withdraw` selector. This path has only been exercised on an Anvil fork, never end to end on mainnet. |
| Fast withdrawals and transfers to another owner need an ECDSA `personal_sign` from the owner. A Safe cannot produce one. | Multi-signature wallets doc | Unchanged from today: secure withdrawal only. |
| Nonces are tracked per API key. Two signers sharing one key would race on nonces. | Core concepts doc | A second reason the executor must have its own slot on any sub-account it signs for. |
| The `SignerClient` in the pinned SDK fork is bound to one `account_index` and can hold several keys for that account. | `lighter/signer_client.py` line 339 | The executor needs one client per account it signs for. |

### Recommended custody model

```text
Lagoon Safe (Ethereum L1)
  │  L1 deposit / withdrawPendingBalance via GuardV0 (existing)
  ▼
Lighter master account        ← executor key (slot 4, exists today)
  │  L2 transfer, master key          ▲  L2 transfer, sub key  — or —
  ▼                                    │  L1 withdraw(subIndex) via guard, no key
Sub-account 1  (Nautilus key slot 4, executor key slot 5)
Sub-account 2  (Nautilus key slot 4, executor key slot 5)
Sub-account N  ...
```

- The master account becomes a **treasury hub**. It holds the Safe-facing
  balance and never trades.
- Each sub-account has two keys: slot 4 for Nautilus, slot 5 for the executor.
  The executor slot is optional if sub-account egress uses the L1 `withdraw`
  path, but it is the only way to move capital **between** sub-accounts
  without a 20-minute round trip through the Safe.
- Lighter API keys have no permission scoping beyond "maker-only", so the
  executor's sub-account keys can also trade. That is the same trust level
  the executor already has on the master key.
- All sub-accounts should be created and their keys registered at deployment
  time, because key registration is a Safe-owner transaction. Creating a
  sub-account later means a new governance transaction.

## Guard smart contract

Files: `contracts/guard/src/GuardV0Base.sol` (Lighter block at lines
839-845 and 1154-1172) and `contracts/guard/src/lib/LighterLib.sol` in
`web3-ethereum-defi`.

| Area | Status now | Updates needed | Complexity |
|------|------------|----------------|------------|
| Whitelisted selectors | `deposit`, `withdraw`, `withdrawPendingBalance` on `ZkLighter` plus USDC `approve`. `changePubKey` deliberately excluded. | None for the recommended model. All sub-account funding and draining is either an off-chain L2 transaction or one of the three existing selectors. | None |
| `deposit` receiver check | `_to` must be an allowed receiver, so only the Safe's master account can be credited. | None. L1 deposits cannot target a sub-account anyway. | None |
| `withdraw(accountIndex, ...)` | Asset index checked. Account index is **unbound**; the comment says the protocol binds `msg.sender` and the only L1 egress is to the Safe. | Still true for sub-accounts: the pending balance lands on the master's L1 address. Optionally add an `allowedAccountIndices` set so a compromised asset manager cannot yank margin out of a live Nautilus sub-account and force liquidations. This is a loss-of-value vector, not an exfiltration vector. | Low (optional) |
| `withdrawPendingBalance` | `_owner` must be an allowed receiver. | None. | None |
| `changePubKey` | Not whitelisted. Key rotation is a Safe-owner transaction. | Keep as is. Whitelisting it with an account-index and slot policy (sub-account indices only, slot ≥ 5) would let the executor hot wallet register keys on new sub-accounts without governance, but a compromised hot wallet could then register its own key and trade the vault's capital. Recommend not doing this; batch all registrations into the deployment governance transaction instead. | Medium if done; recommend skipping |
| Library storage and ABI | `LighterLib` diamond storage holds `allowedContracts` and `allowedAssetIndices`. Compiled ABIs under `eth_defi/abi/guard/`. | Only if the optional account-index allowlist is added: new storage mapping, `allowLighterAccountIndex()` setter, event, `forge build`, ABI refresh, fork tests in `tests/guard/test_guard_lighter_lagoon.py`. | Low |
| Tests | Fork tests cover validation, exfiltration, deposit execution and `withdraw` for a foreign index reverting in the protocol. | Add a fork test that `withdraw(subAccountIndex)` passes the guard. The sequencer-side ownership check cannot be simulated on a fork; it needs the mainnet test described below. | Low |

Conclusion: the guard needs **no mandatory change** for multiple sub-accounts
on the Ethereum deployment.

## web3-ethereum-defi deployment scripts and support functions

Files: `eth_defi/lighter/*.py`,
`eth_defi/erc_4626/vault_protocol/lagoon/deployment.py`,
`scripts/lagoon/lagoon-lighter-*.py`, `scripts/lighter/lagoon-lighter-change-pubkey.py`.

| Area | Status now | Updates needed | Complexity |
|------|------------|----------------|------------|
| Account discovery (`api.py`) | `fetch_lighter_account_index()` returns `min(index)` from `accountsByL1Address` and ignores the rest. `wait_for_lighter_account()` builds on it. | Add `fetch_lighter_accounts(session, l1_address)` returning the master and all sub-account indices with their equity fields in one response, keep the old helper as "master only". Add `wait_for_lighter_sub_account(session, master_index, expected_count)` for post-creation polling. | Low |
| Valuation (`valuation.py`) | `fetch_lighter_total_equity(session, account_index)` reads one account. | Add `fetch_lighter_accounts_equity(session, l1_address)` that returns `dict[int, LighterEquity]` from one `account` by-L1-address response, so NAV across accounts is a single consistent snapshot and an in-flight L2 transfer is not double counted. | Low |
| Deployment config (`LagoonConfig`, `LighterAccountSetup`) | `generate_lighter_api_key: bool`, `lighter_api_key_index: int = 4`. `LighterAccountSetup` is one frozen dataclass with one `private_key`. `LagoonAutomatedDeployment.lighter_account_setup` is singular and is serialised to the operator record JSON. | Add `lighter_sub_account_count: int`, `lighter_sub_account_api_key_index: int = 4` (Nautilus), `lighter_executor_api_key_index: int | None = 5`. Replace the singular setup with `LighterAccountSetup` for the master plus `list[LighterSubAccountSetup]`, each with its index, the Nautilus key and the optional executor key. Keep the JSON shape backwards compatible for existing single-account deployments. | Medium |
| Activation flow (`_activate_lighter_account`) | Subscribes 20 USDC, deposits 1 USDC, waits for the master index, generates one key, `execute_change_pubkey` on the single-owner Safe, waits for the key. | After the master key is live: sign `CreateSubAccount` N times with the master key, poll `accountsByL1Address` until N sub-accounts appear, generate 1 or 2 keys per sub-account, send one `changePubKey` per key through the Safe (ideally batched with MultiSend), wait for each key. Each sub-account also needs a minimum balance before Nautilus can trade; decide whether activation seeds 1 USDC each by L2 transfer. | Medium |
| L2 signing dependency | `lighter-sdk` is deliberately **not** an eth_defi dependency (`docs/superpowers/plans/2026-09-10-lagoon-lighter-api-key-deployment.md` line 184). Only key generation is native. | Sub-account creation and L2 transfers need the signer. Either add `lighter-sdk` as an optional extra (`lighter` extra, pinned fork), or keep eth_defi SDK-free and let trade-executor orchestrate sub-account creation after deployment. The second option splits the deployment into two commands. Recommendation: optional extra with a lazy import, matching the existing lazy import in `lagoon-lighter-trade-example.py`. | Medium |
| L2 transfer helper | None. | Add `transfer_usdc_between_lighter_accounts(signer_client, to_account_index, amount, fee)` wrapping `transfer_same_master_account(...)`, plus `fetch_lighter_transfer_fee(session)` for `transferFeeInfo` and `wait_for_lighter_transfer(session, account_index, tx_hash)` polling `transfer_history`. | Low |
| L1 withdraw helper | No Python helper for `ZkLighter.withdraw()` even though the guard allows it. | Add `request_usdc_withdrawal_from_lighter_account(web3, hot_wallet, vault, account_index, amount)` through `TradingStrategyModuleV0.performCall`. This is the key-free path to pull capital from a sub-account. Needs a mainnet verification that the sequencer accepts a sub-account index with the Safe as master. | Low to write, needs real-money test |
| Key registration helpers (`pubkey.py`) | `encode_change_pubkey`, `build_change_pubkey_safe_tx`, `execute_change_pubkey` already take an arbitrary `account_index`. | Add a batched builder returning a MultiSend payload for many `(account_index, api_key_index, pubkey)` tuples, so a multi-owner Safe signs once. | Low |
| Guard whitelisting (`setup_lighter_whitelisting`) | Whitelists `ZkLighter`, USDC and `allowReceiver(safe)`. | None, unless the optional account-index allowlist is added to the guard. | None |
| Scripts | `lagoon-lighter-change-pubkey.py` prints one Safe transaction. Examples deploy with one key. | Add `lagoon-lighter-create-subaccounts.py` (sign `CreateSubAccount` with the master key, print resulting indices) and extend the change-pubkey script to accept a list. Update `README-lighter-guard.md`. | Low |
| Tests | Mocked API tests select the lowest index; fork tests for guard and `changePubKey`; live tests need `LIGHTER_TEST_ACCOUNT_INDEX`. | Mocked tests for multi-account discovery and equity; fork test for a batched `changePubKey` from the Safe and for guard acceptance of `withdraw(subIndex)`; a FakeSignerClient test for the transfer helper. | Low |

## trade-executor deployment and decide_trades() loop

Files: `tradeexecutor/exchange_account/lighter.py`, `lighter_operator.py`,
`cash_manager.py`, `state.py`, `tradeexecutor/ethereum/lighter/lighter_routing.py`,
`transfer_verification.py`, `tradeexecutor/ethereum/ethereum_protocol_adapters.py`,
`tradeexecutor/cli/commands/*lighter*.py`, `strategies/lighter-ai.py`.

### Hard single-account assumptions

Five places currently enforce exactly one Lighter account per executor:

1. `validate_lighter_exchange_account_pairs()` raises on more than one pair.
2. `create_lighter_exchange_account_pair()` gives every Lighter pair the same
   `pool_address`, `exchange_address` and `internal_id = 1`, so two pairs
   compare equal and collide in position lookup.
3. `create_lighter_vault_valuation_func()` and `_auto_discover_lighter()`
   value NAV as Safe USDC plus `pairs[0]`'s equity.
4. `LighterRoutingConfig.operator_record` holds one account and one key, and
   `_prepare_withdrawal()` never checks that the operator's account matches
   the trade's pair.
5. All CLI options, the operator record JSON, `lighter-move-funds`,
   `lagoon-lighter-test-trade` and `lagoon-redeem` are single valued.

### Component table

| Area | Status now | Updates needed | Complexity |
|------|------------|----------------|------------|
| Pair identity (`lighter.py`, `identifier.py`) | Account index only in `other_data["exchange_subaccount_id"]`; identity fields identical for all Lighter pairs. | Encode the account index into `pool_address`/`exchange_address` and `internal_id`, as Derive does with `hex(subaccount_id)`. Add `other_data` keys for the role (`master` or `sub_account`), the master index and a Nautilus strategy label. Remove the single-pair check, keep the single-protocol check. Existing state files hold the old identity, so a state migration or a compatibility shim for the master pair is required. | Medium |
| Universe construction (`lighter-ai.py`, `create_trading_universe`) | Reads `LIGHTER_ACCOUNT_INDEX` from the environment and builds one pair. | Build one pair per account. Either list indices in `LIGHTER_ACCOUNT_INDEXES`, or discover them from `accountsByL1Address` for the Safe address at universe creation and pin them in the deployment artefact. Discovery is safer against typos but adds a network call in universe construction. | Low |
| NAV (`create_lighter_vault_valuation_func`, `_auto_discover_lighter`, `runner.py` wiring) | Safe USDC plus one account's `total_asset_value`. | Safe USDC plus the sum over all accounts from one by-L1-address snapshot. Fail closed if any account is missing from the response. The existing per-position `ExchangeAccountSyncModel` loop already handles many positions. | Low |
| Operator credentials (`lighter_operator.py`, duplicate in `lagoon_lighter_test_trade.py`) | `LighterOperatorRecord` is one account, one key. Loader reads `deployments.ethereum.lighter_account_setup`. Two diverging copies of the class exist. | Introduce `LighterOperatorRecords` keyed by account index: master key plus optional executor key per sub-account. Load the new JSON shape and the legacy one. Delete the duplicate class in the CLI module. Build one `SignerClient` per account. Add a startup assertion that every pair in the universe that the strategy may withdraw from has a signer or that the L1 `withdraw` path is enabled. | Medium |
| Safe ↔ master transfers (`LighterRouting._prepare_deposit`, `_prepare_withdrawal`) | Deposit: Safe approves and calls `deposit(safe, ...)`, verified by master collateral increase. Withdrawal: SDK secure withdrawal with the operator key, blocking wait, then Safe claim. | Restrict to the master pair and reject a Safe-facing trade on a sub-account pair. Add the account-index cross-check against the operator record. | Low |
| Master ↔ sub-account and sub ↔ sub transfers | Do not exist. | New transfer kind. Model as a pair of linked `TradeExecution`s (negative quantity on the source position, positive on the destination, shared `lighter_transfer_id` in `other_data`), flagged as an internal exchange-account transfer so no reserve allocation, Safe transaction or blockchain receipt is expected. Routing signs `transfer_same_master_account` with the source account's key, records the L2 tx hash, polls `transfer_history`, then verifies source and destination balances. Alternative egress from a sub-account without its key: L1 `withdraw(subIndex)` via the guard, which lands in the Safe after the secure delay and then re-enters as a normal Safe deposit. | High |
| Transfer verification and reconciliation (`transfer_verification.py`, `check-accounts`, `correct-accounts`, `repair`) | Verifies a deposit by collateral delta on the pair's account and a withdrawal by Safe balance delta. Reconciles all pending Lighter transfers. | Add verification for internal transfers using `transfer_history` plus balance snapshots on both accounts; make the fee explicit in accounting (charged on top of the amount). Extend the interrupted-transfer rules so a sub-account transfer that completed on Lighter can be reconciled without a Safe transaction. | Medium |
| Pending-transfer gating (`State.check_if_clean`, `is_external_account_transfer_pending`) | Any pending transfer blocks the whole executor; `decide_trades()` must not emit a new transfer while one is pending. | Internal L2 transfers settle in seconds, so keep the sequential model but allow several internal transfers in one cycle, executed in order. Safe-facing withdrawals keep the blocking 20-30 minute wait. Gate per account rather than globally only if a later version needs concurrency. | Medium |
| Cash policy (`cash_manager.py`) | `ExchangeCashSnapshot` has one scalar exchange balance; `decide()` returns one Safe-facing movement. | Two-layer policy. Layer 1 keeps the existing Safe ↔ master decision, where the master balance is the hub. Layer 2 is a new `SubAccountAllocator` that takes per-account equity and free collateral plus target weights, and returns a list of master → sub and sub → master movements with a minimum transfer size and hysteresis band. Respect that only free collateral can leave a sub-account with open positions; margin in use must not be withdrawn. Keep both layers pure and unit tested. | Medium |
| `decide_trades()` and `PositionManager` | Example strategy uses `get_single_pair()`, `get_current_cash()` and `get_exchange_account_available_balance(pair)`. No helper creates transfers. | Add `PositionManager.rebalance_exchange_accounts(master_pair, target_weights, ...)` or an equivalent helper in `exchange_account/state.py` that builds the linked trade pairs. `lighter-ai.py` gains `Parameters.lighter_sub_account_weights` (static) or a hook for the dynamic allocator output. Opening the N exchange-account positions on first run must stop using "any exchange-account position exists" as the guard. | Medium |
| Backtesting | Lighter positions are value-only; backtests return no transfers. | No change for the first version. | None |
| CLI: `lagoon-deploy-vault`, `deploy_report.py`, `preflight_report.py` | `GENERATE_LIGHTER_API_KEY`, `LIGHTER_API_KEY_INDEX`; reports render one `lighter_account_setup`. | Add `LIGHTER_SUB_ACCOUNT_COUNT`, `LIGHTER_SUB_ACCOUNT_API_KEY_INDEX`, `LIGHTER_EXECUTOR_API_KEY_INDEX`; render the master and each sub-account (indices, key slots, public keys, tx hashes) in the public reports; write the private keys only into the mode-0600 operator record. Produce a separate per-sub-account key file for hand-over to Nautilus so the executor's record never leaves the executor host. | Medium |
| CLI: `start` | `LIGHTER_OPERATOR_RECORD_FILE`, `LIGHTER_WITHDRAWAL_TIMEOUT`, enabled only with `lighter_cash_management`. | Same options, new record shape; add `LIGHTER_TRANSFER_TIMEOUT` for L2 transfers and a flag to prefer L1 `withdraw` over sub-account keys. | Low |
| CLI: `lighter-move-funds`, `lagoon-lighter-test-trade`, `lagoon-redeem` | Each requires exactly one Lighter position and uses the single operator record. | Add `--lighter-account-index` selection, support deposit to and withdrawal from any account, and an internal transfer mode for the manual diagnostic. `lagoon-redeem` must sum all accounts. | Medium |
| `check-accounts`, `correct-accounts` | Loop over exchange-account positions already; Safe check if any Lighter position exists. | Mostly works once pairs are distinct. Add the internal-transfer reconciliation above and a check that the set of on-exchange accounts equals the set of positions. | Low |
| Docs and plans | `README-Lighter.md`, `docs/lagoon-lighter-deployment.md` and all plan docs state one account per vault. | Rewrite the custody model section, deployment steps, env var tables and the Nautilus hand-over procedure. | Low |
| Tests | Fixed-fork Lighter mock serves one account index 126. | Extend the mock to serve a master plus sub-accounts, `transfer_history`, `transferFeeInfo` and `sendTx` for transfers. Cover: NAV across accounts, master → sub transfer, sub → master transfer, interrupted internal transfer reconciliation, and the startup abort on a pending transfer. | Medium |

### Suggested delivery order

1. eth_defi: multi-account discovery and equity helpers, transfer and fee
   helpers, batched `changePubKey` builder, optional `lighter` extra.
2. trade-executor: distinct pair identity with state migration, NAV over all
   accounts, multi-account operator record. At this point a vault can
   **value** many accounts and Nautilus can trade them, with capital moved
   manually.
3. eth_defi deployment: sub-account creation and key registration during
   `lagoon-deploy-vault`.
4. trade-executor: internal transfer trade kind, routing, verification,
   reconciliation and the allocator policy; `decide_trades()` helper.
5. CLI diagnostics and docs.

Mainnet verification needed before step 4 is finalised: one real
`CreateSubAccount` from a Safe-owned master, one `changePubKey(subIndex)` from
the Safe, one L2 transfer in each direction, and one L1 `withdraw(subIndex)`
through the guard.

## Lighter EVM deployment

Lighter EVM is an EVM-equivalent rollup built with Axiom that settles together
with Lighter Core. Status on 2026-10-26: **devnet only**, opened on
2026-10-16, access by request to `evm@lighter.xyz`, private L1 for settlement,
mocked ZK proofs, escape hatch not functional. Chain id `19529`
(`0x4C49`), gas paid in ETH. No mainnet date is published.

What the interop design means for this work, from the Lighter EVM docs:

| Capability | Lighter EVM mechanism | Effect on the multi-subaccount design |
|------------|----------------------|----------------------------------------|
| Account ownership | Core treats `(Lighter EVM, address)` as a controller distinct from `(Ethereum L1, address)`. A Safe deployed on Lighter EVM gets its **own** master account on first deposit; its L1 twin is a different account. | A Lagoon vault on Lighter EVM is a fresh custody, not a bridge to the existing Ethereum-owned account. |
| Deposit | `EvmInteropManager.sendToCore(assetIndex, to, isL1Account, amount, routeType)`; master accounts only, "transfers to Core subaccounts are not supported". | Same two-step funding as L1: deposit to master, then an L2 transfer with the master key. |
| Withdraw | `EvmInteropManager.sendToEvm(accountIndex, assetIndex, evmRecipient, routeType, baseAmount)`; `accountIndex` may be the master **or any sub-account**; authorised by `msg.sender` being the controlling address. No API key, no secure-withdrawal delay documented. | The executor would need **no sub-account keys at all** for egress. The whole "two keys per sub-account" problem disappears; only the master key for master → sub transfers remains. |
| Key registration | `EvmToCoreEntrypoint.changePubKey(accountIndex, apiKeyIndex, pubKey)` at predeploy `0x66…09`, callable by the controlling contract. | Same shape as L1, so the batched Safe transaction design carries over. |
| Sub-account creation | Not listed among the entrypoint functions. Presumably still an L2 `CreateSubAccount` signed with the master key. | Unchanged; needs confirmation once docs are complete. |
| Trading from contracts | `createOrder`, `cancelOrder`, `cancelAllOrders` on the entrypoint; writes are asynchronous with a `reqId` and a 14-day escape hatch. | Not needed for this work. Nautilus keeps using API keys against the same Core. |
| Reading Core state | `READ_CORE` precompile at `0x66…00` for accounts, balances, positions and request status. | On-chain NAV becomes possible, which would let the Lagoon valuation be computed without the off-chain API. |
| Failure modes | A reverted delivery is not retried; for `sendToCore` the unaccepted amount "is lost"; for Core → EVM the debit is not restored. | Transfer verification must check `acceptedAmount` and request status, not only the EVM receipt. |

Gap table for a Lagoon vault on Lighter EVM:

| Area | Status now | Updates needed | Complexity |
|------|------------|----------------|------------|
| Chain availability | Devnet, gated, mocked proofs. | Wait for public testnet and mainnet. Nothing production-grade can be built yet. | Blocked |
| Lagoon and Safe contracts | Lagoon factory and Safe singleton exist on Ethereum, Base, Arbitrum and the other chains eth_defi supports. Nothing on chain 19529. | Lagoon team must deploy the protocol; Safe contracts must be deployed; eth_defi `LAGOON_FACTORY`-style constants, chain config, USDC address and RPC middleware for chain 19529 added. | High, external dependency |
| Guard | `LighterLib` validates the Ethereum `ZkLighter` selectors only. | New `LighterEvmLib` (mirroring `HypercoreVaultLib`, the HyperEVM analogue) covering `sendToCore` (`to` must be the Safe, asset allowlist), `sendToEvm` (`evmRecipient` must be the Safe), and optionally `changePubKey` with a sub-account-only policy. Library linking, ABI refresh, fork tests against the devnet predeploys. | High |
| eth_defi helpers | Ethereum-only session, contract addresses and whitelisting. | `LighterDeployment.create_lighter_evm()`, interop manager ABIs, `accountsByEvmAddress` discovery, request-status reads via `READ_CORE`, deposit and withdraw helpers through the module. | Medium |
| trade-executor routing | `LighterRouting` hard-codes the `ZkLighter` flow and the SDK secure withdrawal. | A second routing implementation keyed by deployment, selected from the pair's `exchange_deployment` metadata. Sub-account egress becomes a single Safe transaction. | Medium |
| Investor side | Lagoon investors deposit USDC on Ethereum today. | Investors would need USDC on Lighter EVM (L1-native USDC appears as a wrapper token minted by the gateway). Front-end and bridge UX are outside this repository. | High, external |

Recommendation: design the Ethereum multi-subaccount work so that the custody
layer (deposit, withdraw, transfer) is behind a per-deployment routing
interface, and treat Lighter EVM as a second deployment of the same abstract
model once it reaches mainnet. Do not block the Ethereum work on it.

## Open questions

1. Does the Lighter sequencer accept `CreateSubAccount` and sub-account
   `changePubKey` for a Safe-owned master in practice? The docs say yes, the
   contract forwards both; only a mainnet test proves it.
2. Does the L1 `withdraw(subIndex)` priority request succeed for a sub-account
   with open positions when only free collateral is requested, and what is the
   observed delay compared with the API secure withdrawal?
3. Should the executor hold a key on each sub-account (fast sub ↔ sub moves,
   more secrets to protect) or rely on L1 `withdraw` (no extra secrets, slow)?
   The recommendation above is both: register slot 5 at deployment, use the
   L2 path by default, keep L1 `withdraw` as the recovery path.
4. How does Nautilus learn its sub-account index and key? The deployment
   should write one hand-over file per sub-account.
5. Which account tier does production need? Standard allows 4 sub-accounts;
   Plus allows 16 with a 0.005% fee.
6. Which state migration strategy is used for the existing `lighter-ai`
   production state when the master pair's identity changes?

## References

- `eth_defi/lighter/README-lighter-guard.md` (guard scope and security model)
- `contracts/guard/src/lib/LighterLib.sol`, `contracts/guard/src/GuardV0Base.sol`
- `eth_defi/erc_4626/vault_protocol/lagoon/deployment.py` (`LighterAccountSetup`, `_activate_lighter_account`)
- `tradeexecutor/exchange_account/README-Lighter.md`, `docs/lagoon-lighter-deployment.md`
- `docs/plans/lagoon-lighter-api-key-deployment.md` (the explicit one-account decision)
- `docs/plans/lighter-cash-management-simplification.md` (current cash-management design)
- Lighter docs: core concepts, API keys, deposits/transfers/withdrawals,
  multi-signature wallets, account types, priority transactions, Lighter EVM
  interop (accounts, token transfers, message passing), all under
  <https://docs.lighter.xyz/>
- Verified `ZkLighter` implementation `0x831ef69bab8af8b1037a4961b8d0674b124e7008`
- Lighter EVM devnet announcement, 2026-10-16
