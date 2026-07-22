# eth-defi vault test follow-up instructions

## Purpose

This document is a hand-off for agents working in `web3-ethereum-defi` after
the first complete `vault-test-trade --auto-simulated` run. Treat each numbered
work item as a separate, reviewable change. Do not combine unrelated protocol
adapters, CCTP support and RPC reliability changes in one pull request.

The run tested 129 explicitly selected vaults across Ethereum, Base, Arbitrum,
Avalanche and Monad. The final result was:

| Result | Count |
|---|---:|
| Successful simulated deposit and redemption | 36 |
| Deposit closed | 20 |
| Asynchronous simulation unsupported | 28 |
| Failed | 45 |

Not every failure means that a protocol adapter is missing. The list below
separates genuine adapter gaps from vault admission rules, closed or unhealthy
vaults, and RPC/fork failures.

The infrastructure-only subset was rerun on 2026-07-22 after trade-executor
gained disposable Anvil generations. Twelve of the fourteen Base vaults passed;
the remaining Gains and IPOR failures became deterministic protocol results.
The Ethereum Muchacho timeout likewise became a deterministic Lagoon `XJy8`
revert. Section 4 retains the original IDs as historical evidence but records
their current classification.

## Working locations and required reading

The populated eth-defi checkout is:

- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi`

Do not use the empty submodule directory under the trade-executor feature
worktree. Before changing code, read the instructions in the eth-defi checkout,
then read these shared abstractions completely:

- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/AGENTS.md`, if present
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/vault/deposit_redeem.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/classification.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/deposit_redeem.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/analysis.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/flow.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/settlement_scan.py`

The consumer-side flow and its design notes are:

- `/home/mikko/code/trade-executor-vault-test-trade/.claude/docs/vault-deposit-redeem.md`
- `/home/mikko/code/trade-executor-vault-test-trade/tradeexecutor/cli/testtrade.py`
- `/home/mikko/code/trade-executor-vault-test-trade/tradeexecutor/ethereum/vault/vault_routing.py`
- `/home/mikko/code/trade-executor-vault-test-trade/tradeexecutor/cli/vault_test_trade.py`
- `/home/mikko/code/trade-executor-vault-test-trade/tradeexecutor/cli/commands/vault_test_trade.py`
- `/home/mikko/code/trade-executor-vault-test-trade/tests/lagoon/test_lagoon_e2e.py`

The source universe and captured evidence are:

- `/home/mikko/code/getting-started/scratchpad/xchain2/20-backtest-ignore-missing-fee-vaults-cleaned.ipynb`
- `/tmp/vault-test-notebook-final-lET3iF/final-results.md`
- `/tmp/vault-test-notebook-final-lET3iF/final-table.log`
- `/tmp/vault-test-notebook-final-lET3iF/vault-test-notebook.json`

The `/tmp` paths are diagnostic artefacts from the 2026-07-22 run and may be
removed by the operating system. Relevant vault IDs and failure signatures are
therefore repeated below.

## Unsupported or incomplete vault protocol support

### Summary

| Priority | Protocol or behaviour | Affected vaults | Gap |
|---|---|---:|---|
| P0 | Upshift multi-asset | 1 | Explicitly unsupported deposit flow |
| P0 | Plutus ERC-7540 | 1 | Generic pricing calls unsupported `previewDeposit()` |
| P0 | Accountable ERC-7540 | 1 | Generic pricing calls unsupported `previewDeposit()` |
| P0 | Ember | 5 | Deposits succeed, but receipt analysis rejects their events |
| P0 | Gains Network | 1 | Redemption broadcasts, but its transaction/result type cannot be analysed |
| P1 | cSigma Finance | 2 | Restricted/partial redemption is not modelled correctly |
| P1 | Yearn | 1 confirmed | Successful redemption event layout is not analysed correctly |
| P1 | YieldNest | 1 confirmed | Successful deposit event layout is not analysed correctly |
| P1 | D2 Finance | 1 | Zero/undefined initial share price raises `DivisionByZero` |
| Separate workstream | Lagoon/ERC-7540 simulation | 28 | No Anvil-only async request, settle and claim simulation API |

IPOR Fusion and eight Lagoon `XJy8` failures are not yet proven adapter gaps.
Investigate their admission rules and decode the errors before changing an
adapter. The Base timeouts and empty historical-state responses from the first
run were provider or fork issues, not protocol support issues; the focused
rerun results are recorded in section 4.

### 1. Upshift multi-asset deposits

Affected vault:

- `1-0x74ad2f789ed583dbd141bbdafc673fe1f033718b` — Sentora USD Earn

Observed failure:

`Upshift multi-asset vault deposits are not supported by the generic ERC-4626 deposit manager`

Read:

- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/upshift/vault.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/erc_4626/vault_protocol/test_upshift.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/erc_4626/test_upshift_multi_asset_events.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/erc_4626/test_fix_upshift_vaults_script.py`

Instructions:

1. Determine the accepted input asset and conversion/queue contract for this
   concrete vault from on-chain calls and verified contract sources. Do not
   assume that ERC-4626 `asset()` is sufficient for a multi-asset Upshift vault.
2. Add an `UpshiftDepositManager` under
   `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/upshift/`.
   It must implement request construction, capability queries, estimation and
   transaction analysis using the real Upshift entry point and events.
3. Return the protocol-specific manager from `UpshiftVault.get_deposit_manager()`.
   Do not weaken the generic ERC-4626 manager to guess multi-asset semantics.
4. Add a mainnet-fork test for the affected vault covering deposit and every
   redemption phase it supports. Assert asset spent, shares received, shares
   burnt and asset returned using decoded events and balance changes.
5. If the vault requires an allowlist or a delayed claim, expose that through
   `can_create_*`, `has_synchronous_*`, tickets and `can_finish_*`; do not hide
   it behind retries.

Completion means `Vault.get_deposit_manager()` can execute and analyse the
vault without any Upshift knowledge in trade-executor.

### 2. Plutus ERC-7540 support

Affected vault:

- `42161-0x58bfc95a864e18e8f3041d2fcd3418f48393fe6a` — Plutus Hedge Token

Observed failure:

`previewDeposit() is not supported for ERC-7540 vaults`

Read:

- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/plutus/vault.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/erc_4626/vault_protocol/test_plutus.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/vault/deposit_redeem.py`

Instructions:

1. Confirm whether this Plutus generation follows ERC-7540 request/claim or a
   Plutus-specific queue. Record the exact callable functions, events and
   lifecycle in the adapter docstring and test.
2. Implement a Plutus deposit manager instead of routing pricing through the
   generic ERC-4626 `previewDeposit()` path.
3. Supply a valid estimate through protocol state when one exists. If no exact
   preview exists, expose an explicit capability or conservative estimate that
   does not divide by a fabricated value. Never pretend the request is
   synchronous.
4. Represent pending deposits and redemptions with `DepositTicket` and
   `RedemptionTicket`, including enough identifiers to resume from persisted
   state.
5. Extend the Arbitrum fork test to exercise request, readiness detection,
   claim and event analysis. If settlement needs a privileged actor, impersonate
   that actor only in the test and explain why.

### 3. Accountable ERC-7540 support

Affected vault:

- `143-0x7cd231120a60f500887444a9baf5e1bd753a5e59` — Hyperithm Delta Neutral Vault

Observed failure:

`previewDeposit() is not supported for ERC-7540 vaults`

Read:

- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/accountable/vault.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/accountable/deposit_redeem.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/accountable/settlement.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/erc_4626/vault_protocol/test_accountable.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/erc_4626/vault_protocol/test_accountable_settlement.py`

The protocol already has specialised code, so first trace why estimation still
reaches generic `previewDeposit()`. Fix adapter selection or the Accountable
manager's estimate methods at the protocol boundary. Add a Monad fork regression
for the exact vault. Preserve its true asynchronous lifecycle and do not add a
generic ERC-7540 special case to trade-executor.

### 4. Ember receipt analysis

Affected vaults:

- `1-0x2b13311fd553e74b421d4ccc96e348f71e179dcf` — Ember Apollo ACRED
- `1-0x9be9294722f8aad37b11a9792be2c782182cafa2` — Ember Earn
- `1-0x0b9342c15143e8f54a83f887c280a922f4c48771` — Ember Polymarket
- `1-0xf3190a3ecc109f88e7947b849b281918c798a0c4` — Ember Third Eye
- `1-0x373152feef81cc59502da2c8de877b3d5ae2e342` — Ember UDL

All five deposit transactions had receipt status `1`, transferred USDC and
minted shares. `analyse_4626_flow_transaction()` nevertheless returned a
failure because Ember emits a custom flow around the standard transfers.

Read:

- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/ember/vault.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/ember/deposit_redeem.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/ember/settlement.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/erc_4626/vault_protocol/test_ember_deposit_redeem.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/erc_4626/vault_protocol/test_ember_settlement.py`

Instructions:

1. Decode the exact successful receipts from the state diagnostic or reproduce
   one vault on an Ethereum fork. The receipt contains the standard ERC-20
   asset transfer and share mint plus Ember-specific events.
2. Implement `analyse_deposit()` and, if necessary, `analyse_redeem()` on the
   Ember manager. Filter by the concrete vault, owner and receiver; do not use
   the first `Transfer` event in a receipt.
3. Prefer protocol events as the source of truth. Use before/after balances only
   as a test cross-check, not as production analysis that would require replay.
4. Add one regression using the exact production vault and a parametrised
   classification/receipt test for the remaining addresses.
5. Return `DepositRedeemEventAnalysis` with the actual raw-to-decimal asset and
   share quantities.

### 5. Gains Network redemption analysis

Affected vault:

- `42161-0xd3443ee1e91af28e5fb858fbd0d72a63ba8046e0` — gTrade USDC

Observed failure:

The redemption transaction was broadcast, after which the caller raised
`Unimplemented swap transaction type`.

Read:

- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/gains/vault.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/gains/deposit_redeem.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/gains/README-Ostium.md`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/gains/test_gtrade_usdc.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/gains/test_ostium.py`

Instructions:

1. Reproduce the exact redemption and identify which request, ticket or
   analysis result escapes the `VaultDepositManager` contract expected by the
   caller.
2. Make the Gains manager return the same abstract request/ticket and
   `DepositRedeemEventAnalysis` types as other vault managers. Protocol-specific
   transaction functions may remain internal to the adapter.
3. Correctly model any unlock epoch or claim step. A broadcast request is not a
   completed redemption.
4. Add an Arbitrum fork test for a complete deposit/redemption lifecycle and a
   resume test that reconstructs readiness from a persisted ticket.

### 6. cSigma restricted redemption

Affected vaults:

- `1-0xd5d097f278a735d0a3c609deee71234cac14b47e` — cSigma USD
- `1-0x438982ea288763370946625fd76c2508ee1fb229` — cSuperior Quality Private Credit USDC

The first vault allowed only about 5% of held shares through `maxRedeem()`;
the test attempted to redeem the whole position. The second completed a deposit
but its sell path failed without a useful protocol-level diagnostic.

Read:

- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/csigma/vault.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/erc_4626/vault_protocol/test_csigma.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/flow.py`

Determine whether partial redemption is normal throttling, a per-epoch limit or
a pending queue. The adapter must expose the lifecycle honestly. If partial
instant redemption is supported, create a request for at most `maxRedeem()` and
leave the remainder open. If it is an asynchronous queue, add tickets and claim
handling. Never bypass `maxRedeem()` in production code. Add tests for both
partial progress and a zero-current-capacity result.

### 7. Yearn and YieldNest event variants

Affected vaults:

- `1-0x33ffc177a7278ff84aab314a036bc7b799b7cc15` — Yearn Arche USD; redeem status was `1`, but analysis failed
- `1-0x01ba69727e2860b37bc1a2bd56999c1afb4c15d8` — YieldNest RWA MAX; deposit status was `1`, but analysis failed

Read:

- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/yearn/vault.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/yieldnest/vault.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/erc_4626/vault_protocol/test_yearn_yvault.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/erc_4626/vault_protocol/test_yieldnest.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/analysis.py`

Reproduce and decode each successful receipt. If the event layout is a valid
protocol-wide variant, add protocol-specific analysis. Change the generic
ERC-4626 analyser only if the receipt is standards-compliant and the new logic
remains strict about vault, owner, receiver and direction. A status-1 receipt
alone is not enough to infer amounts. Add fork regressions for both exact
addresses.

### 8. D2 zero or undefined share price

Affected vault:

- `42161-0x75288264fdfea8ce68e6d852696ab1ce2f3e5004` — HYPE++

Observed failure: the buy-price path raised `decimal.DivisionByZero`.

Read:

- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/d2/vault.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/d2/settlement.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/erc_4626/vault_protocol/test_d2.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/erc_4626/vault_protocol/test_d2_settlement.py`

Inspect total assets, total supply, preview calls, deposit caps and protocol
state at the fork block. If this is an empty or not-yet-open vault, return an
explicit capability/admission result rather than a numerical price. If D2 has a
defined initial exchange rate, implement it in the D2 adapter and test both an
empty vault and a live vault. Do not turn division by zero into an arbitrary
one-to-one price.

## Avalanche CCTP V2 support

### Scope

Mainnet Avalanche C-Chain support is required. It currently blocks these vaults
before their vault adapters are exercised:

- `43114-0x124d00b1ce4453ffc5a5f65ce83af13a7709bac7` — 40acres Pharaoh USDC
- `43114-0x9fd32fd5e32c6b95483d36c5e724c5c5250ce010` — Yearn ygamiUSDC
- `43114-0x7aca67a6856bf532a7b2dea9b20253f08bc9a85a` — Yearn ymevUSDC

Circle's current CCTP V2 documentation assigns Avalanche domain `1`. The V2
TokenMessenger, MessageTransmitter and TokenMinter addresses are the same
CREATE2 addresses already used by eth-defi on Ethereum, Arbitrum and Base:

- `https://developers.circle.com/cctp/concepts/supported-chains-and-domains`
- `https://developers.circle.com/cctp/references/contract-addresses`

Read:

- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/cctp/constants.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/cctp/transfer.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/cctp/bridge.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/cctp/receive.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/cctp/testing.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/cctp/whitelist.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/cctp/README-cctp.md`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/token.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/cctp/test_cctp_dual_fork.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/cctp/test_cctp_transfer_fork.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/cctp/test_cctp_lagoon_fork.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/lagoon/test_lagoon_multichain_deploy.py`

Instructions:

1. Add `CCTP_DOMAIN_AVALANCHE = 1` to
   `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/cctp/constants.py`.
2. Add `43114: CCTP_DOMAIN_AVALANCHE` to `CHAIN_ID_TO_CCTP_DOMAIN` and add
   `CCTP_DOMAIN_AVALANCHE: "Avalanche"` to `CCTP_DOMAIN_NAMES`. The reverse
   mapping is generated from the forward mapping and should then include it.
3. Confirm on the fork that the existing CCTP V2 contract addresses contain
   code on Avalanche. Do not introduce Avalanche-specific contract addresses;
   Circle documents the same V2 addresses for Avalanche.
4. Native Avalanche USDC is already defined as
   `0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E` in
   `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/token.py`.
   Add a known funded Avalanche account to `USDC_WHALE` for fork tests, or use
   a locally scoped, documented holder fixture. Verify the holder at the chosen
   fork block instead of copying an unverified address.
5. Add small unit assertions for domain resolution, reverse resolution and
   display name. Also assert that `prepare_deposit_for_burn(...,
   destination_chain_id=43114)` binds domain `1`.
6. Add an Avalanche mainnet-fork test that burns native USDC through
   TokenMessengerV2 and checks the source balance and emitted CCTP message.
7. Extend a dual-fork test to cover Ethereum to Avalanche and Avalanche to
   Ethereum. In simulation, use `replace_attester_on_fork()`,
   `craft_cctp_message()` and `forge_attestation()`; do not call Circle Iris for
   an Anvil-only test. Test both directions because the vault test must fund a
   satellite and later return redeemed assets.
8. Extend the Lagoon multichain deployment test with an Avalanche fork. Assert
   that the deterministic Safe address matches other chains, the Avalanche
   guard allows CCTP domain `1`, and a forged-attestation bridge arrives as
   native Avalanche USDC.
9. Update `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/cctp/README-cctp.md`
   to list Avalanche and its domain.
10. Avalanche Fuji support is useful but must be a complete addition: chain ID
    `43113`, official testnet native USDC, `TESTNET_CHAIN_IDS`, the testnet
    domain map and a Fuji test. Do not add the chain/domain without its token
    metadata and test coverage.

Acceptance criteria:

- `_resolve_cctp_domain(43114) == 1`.
- CCTP whitelist and Lagoon configuration code picks Avalanche up through the
  shared mappings without caller-side hard-coding.
- Simulated Ethereum ↔ Avalanche transfers pass on dual Anvil forks.
- The three affected Avalanche vault tests reach the vault deposit code rather
  than failing with `Destination chain 43114 is not supported by CCTP`.

## Other issues

### 1. Asynchronous vault simulation

Twenty-eight Lagoon vaults could submit an asynchronous deposit request, but
eth-defi has no reusable Anvil-only facility to drive request → settle → claim
and redeem request → settle → claim for an arbitrary deployed Lagoon vault.
The command therefore records `simulation_unsupported_async`, closes its
simulated position and continues. This is expected until the following larger
workstream is implemented.

Read:

- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/lagoon/deposit_redeem.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/lagoon/settlement.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/lagoon/testing.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/lagoon/analysis.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/lagoon/test_erc_7540_deposit_redeem.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/lagoon/test_lagoon_settlement.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/lagoon/test_lagoon_multichain_deploy.py`
- `/home/mikko/code/trade-executor-vault-test-trade/tests/ethereum/test_lagoon_crosschain_simulated.py`
- `/home/mikko/code/trade-executor-vault-test-trade/tests/mainnet_fork/test_xchain_master_vault_multichain.py`
- `/home/mikko/code/trade-executor-vault-test-trade/tests/lagoon/test_master_chain_multichain_deploy.py`

Add an explicit testing API that is only usable on Anvil. Given a vault,
depositor/receiver and request ticket, it should locate or impersonate the
settlement roles, post the required valuation, settle, advance time/block when
needed and claim. It must exercise the real deployed contracts and return the
ordinary deposit-manager ticket/analysis objects. It must not fake a successful
receipt or add simulation branches to production transaction construction.

Cover both deposit and redemption on a deployed Lagoon fork. Then expose the
same pattern for protocol-specific ERC-7540 managers where settlement is
possible. Until this exists, `simulation_unsupported_async` is the correct
result and must not be converted to success.

### 2. IPOR Fusion admission failures

Six Ethereum IPOR vaults reverted deposits with selector `0x068ca9d8` and an
encoded Safe address:

- `1-0x95b2ed8f821570f85fd0e3e6e7088c6296587088`
- `1-0x888e1d3c509c80e24cab8a4872e164b7e5a6eb10`
- `1-0xc825779c89120eeef746c51130b362478e181d39`
- `1-0x4c5a611694c426cae9335d53e95b885090cf8c31`
- `1-0x32f07401eb177f2c0fc4f95f3928050d88dae7ed`
- `1-0xc2a119ea6de75e4b1451330321cb2474eb8d82d4`

Read:

- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/ipor/vault.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/ipor/README-IPOR-Curators.md`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/erc_4626/vault_protocol/test_ipor.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/ipor/test_ipor_deposit.py`

Decode the selector from the verified ABI/source and reproduce one failing and
one successful IPOR vault with the same owner address type. If it is an
allowlist, private-vault or receiver restriction, add a cheap protocol-specific
admission query and a typed explanation. If no reliable view exists, preserve
the on-chain failure but decode it. Do not call these vaults unsupported and do
not bypass their access control.

### 3. Lagoon admission failures

Eight Ethereum Lagoon vaults reverted deposits with the short reason `XJy8`:

- `1-0x3be67ba2d3fec744d1d2b5d564c83f57372578e4`
- `1-0x9fdbaaa76194d56e49cade12c1f216f47d2b865e`
- `1-0xf10801bcc3deaf467fb8b3dbb7430111822e6dab`
- `1-0xba6cfe8a9d199cd7f3e50114c4e4ec66f2d52c87`
- `1-0xb993c32f578e5156369330787cf8c8fe033bf40e`
- `1-0xcb58582b0d52ce5feecb06ba9ce66598b0d57886`
- `1-0x175ea882b492c9b7a6d5852fe9da560dc7af1c72`
- `1-0xef39d77c7fb6224ac974c5fa4e3151a6c6ce9594`

Read:

- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/lagoon/vault.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/lagoon/deposit_redeem.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/lagoon/lagoon_compatibility.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/lagoon/test_lagoon_erc_4626.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/tests/lagoon/test_lagoon_erc_7540.py`

Decode `XJy8` against the deployed implementation and identify whether each
vault is closed, private, receiver-restricted, capped or incompatible. Improve
the Lagoon capability/admission query where the condition can be checked before
broadcast. Preserve unknown reverts as failures with decoded context. Do not
add automatic retries.

### 4. Base fork reliability and performance

The first run classified the following as infrastructure failures rather than
confirmed adapter gaps:

- Gains: `8453-0xad20523a7dc37babc1cc74897e4977232b3d02e5`
- IPOR: `8453-0x0d877dc7c8fa3ad980dfdb18b48ec9f8768359c4`,
  `8453-0xd6701905c59ee618dc36dc747506bce0a4ac760a`,
  `8453-0x45aa96f0b3188d47a1dafdbefce1db6b37f58216`,
  `8453-0x1166250d1d6b5a1dbb73526257f6bb2bbe235295`
- Yearn timeouts: `8453-0x94d886d25729150adfa20210f9b94cefe0b3d132`,
  `8453-0xcb2f26898c0893c0bdd5cf76417cf9b2258af0ed`,
  `8453-0x50fd1e6e0e2153c2b26ebbcd9bcded4639a1aae3`,
  `8453-0x7bb8b0b176199b3313642dde3421e38e548d6570`,
  `8453-0x19f233b2953275196e6343f17b76da098c478e21`,
  `8453-0xb13cf163d916917d9cd6e836905ca5f12a1def4b`,
  `8453-0xd1468af648565f11393e4033cb0cd270b62495c9`
- Missing historical state at Base block `48,960,873`:
  `8453-0xc3bd0a2193c8f027b82dde3611d18589ef3f62a9` and
  `8453-0x92a6f4cc1e26baa1beec168e3c346aedcb437f31`

The focused rerun used an Ethereum home-chain fork and Base satellite fork. Its
current result is:

- Twelve vaults completed a simulated CCTP deposit and redemption cycle.
- Gains `8453-0xad20523a7dc37babc1cc74897e4977232b3d02e5` reached redemption and then failed
  deterministically because its sell transaction type is not analysed.
- IPOR `8453-0xd6701905c59ee618dc36dc747506bce0a4ac760a` reached redemption and reverted
  deterministically with custom error `0x1425ea42`.
- Teller `8453-0x19f233b2953275196e6343f17b76da098c478e21` encountered one local 40-second
  read timeout. Trade-executor killed both forks, created a fresh multichain
  generation and reran Teller successfully.
- Both former missing-state cases completed successfully, so they are not
  evidence of an adapter defect.

Keep protocol-specific sleeps and same-process Anvil retries out of eth-defi.
Upstream RPC failover belongs in the fallback proxy. If local Anvil stops
responding, the consumer must destroy the complete fork generation and start a
new one; retrying localhost only talks to the same unhealthy process.

### 5. Diagnostics and failure typing

The test exposed several successful on-chain transactions reported only as
generic `Test buy failed`, `Test sell failed` or `Failed to analyse vault tx`.
Improve eth-defi failures so they preserve:

- protocol and vault address;
- request, settlement or claim phase;
- transaction hash and receipt status;
- decoded custom error name and arguments when available;
- whether the transaction reverted or only post-transaction analysis failed.

Read:

- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/vault/deposit_redeem.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/erc_4626/analysis.py`
- `/home/mikko/code/trade-executor/deps/web3-ethereum-defi/eth_defi/revert_reason.py`
- `/home/mikko/code/trade-executor-vault-test-trade/tradeexecutor/cli/testtrade.py`

Keep the public deposit-manager result types stable. Prefer structured failure
data in eth-defi and let callers format it. Never reinterpret a status-1 receipt
as a failed transaction merely because analysis is incomplete.

## Implementation and test rules

For every protocol task:

1. Start with one exact failing production address on an Anvil fork.
2. Use `create_multi_provider_web3()` and
   `wait_for_transaction_receipt_robust()`.
3. Use `TokenDetails.convert_to_raw()` and token transfer helpers; do not
   hard-code decimal multipliers in production code.
4. Keep protocol behaviour inside its vault class and deposit manager. The
   shared `VaultDepositManager` interface is the boundary consumed by
   trade-executor.
5. Test the true capability: synchronous flows finish in one transaction;
   asynchronous flows persist tickets and expose request readiness and claim
   separately.
6. Assert decoded amounts and state transitions, not only receipt status.
7. Do not add protocol-operation retries, same-process Anvil retries, or
   silently turn admission failures into `deposit closed`.
8. Run focused tests with the eth-defi Poetry environment and required RPC
   variables. Keep each individual invocation under ten minutes.

After the focused eth-defi tests pass, rerun only the affected vault IDs through
the trade-executor worktree. The command has no strategy module: it constructs
the trading universe directly from the downloaded `VaultUniverse`, then uses
the eth-defi vault adapter through `perform_test_trade()`.

The end-to-end acceptance condition is that each retested row becomes one of:

- `success (simulated)` for a supported instant lifecycle;
- `simulation unsupported async` only for a real asynchronous lifecycle that
  the simulation helper still cannot settle;
- `deposit closed` or another precise admission result when the vault is not
  currently open;
- `failed` with a decoded, actionable protocol reason; or
- `infrastructure failed` after one complete Anvil-generation replacement also
  fails.

Do not report success merely because a transaction was broadcast. A successful
instant test must analyse both deposit and redemption and finish with the
simulated position closed.
