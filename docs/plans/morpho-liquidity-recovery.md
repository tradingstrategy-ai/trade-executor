# Morpho liquidity recovery

Implement the public recovery demonstrated for Api3 dCOMP USDC and Purinta
USDC at Ethereum block 26402309. Ordinary exact redemption remains the first
choice. The strategy universe and historical backtest admission are out of scope.

1. Add one optional recovery penalty limit to the existing protocol rules.
   Enable a one-basis-point limit only for the two reviewed identities in
   `tradeexecutor/strategy/vault_protocol/morpho.py`. Document the source analysis,
   quantities, oracle controls and the distinction from synthetic Anvil funding.
2. Extend eth-defi's Morpho manager to recognise the demonstrated configured
   exit-market shortfall after `Panic(0x11)`. Use stored ABIs to select a funded
   alternate market in the same adapter and preflight public `forceDeallocate`.
   Refuse unrelated failures, insufficient liquidity and penalties over the cap.
3. Reuse the existing request/finish lifecycle and persisted trade transactions.
   The first receipt proves the penalty shares burned; the ticket retains the
   original requested shares and the remaining shares. Finish redeems the latter.
   Account for the entire original share disposal, actual net cash and one fee.
   A restart resumes the ticket and never repeats the recovery transaction.
   Checkpoint the paid ticket and finish immediately in common EVM receipt
   handling. Track retained wallet shares separately from trading availability,
   so even an unavailable finish passes custody reconciliation correctly.
4. Extend Guard validation with an explicit owner-configured recovery fee cap,
   custody-owner checks and registered adapter checks. Configure this permission
   from the same rule through existing deployment descriptors. Compile artifacts.
5. Let public test-trade commands resolve immediately claimable staged requests
   through the normal resolver before any Anvil-only operator intervention.
6. Validate normal/no-rule behaviour and refusal boundaries, persisted ticket and
   fee accounting, Guard rejection cases and both real fixed-fork recovery cases.
   Preserve evidence under ignored research data. Review the finished patch.

## Sanity check

Checked against the existing manager, routing, settlement retry, state store,
protocol overlay and Guard deployment paths. No separate journal, recovery
service, general market optimiser or new dependency is needed. Request-level
staging does not change Morpho's historical synchronous classification. Recovery
is bounded to one alternate market and one public deallocation; a later failure
remains pending for inspection rather than charging repeated recovery penalties.
The cap limits the configured rate; integer rounding of the actual fee is checked
against the corresponding rounded asset amount in the mined receipt.

Evidence: https://github.com/tradingstrategy-ai/vault-notebooks/pull/25#issuecomment-6896601999
and `notebooks/vault-of-vaults-round-3/morpho-arithmetic.md` in that repository.

Independent Opus 5.5 review identified live-cycle timing and custody-accounting
gaps missed by the original public CLI smoke test. The implementation now
finishes at the common live EVM receipt boundary and tests that boundary against
a real fork, including pending valuation and the production account checker.
Current live execution authorisation also applies to older open-position pairs
and can revoke cached grants without rewriting historical metadata.
