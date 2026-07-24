# Trade-executor cross-chain vault test report

## Run

The complete 129-vault cross-chain simulated matrix was run on 2026-07-24
with `trade-executor` at `248e1e20` and `web3-ethereum-defi` master at
`b5803bdc5`. The invocation used `--auto-simulated --settle-async-on-anvil`.

| Result | Vaults |
|---|---:|
| Deposit closed | 34 |
| Receipt analysis failed | 92 |
| Transaction reverted | 2 |
| Execution failed | 1 |

The only execution-admission gap is Upshift Sentora USD Earn
(`1-0x74ad2f789ed583dbd141bbdafc673fe1f033718b`), whose multi-asset deposit
is explicitly unsupported by the generic ERC-4626 manager.

## Trade-executor fixes needed

1. Preserve a completed deposit and its decoded receipt as success when the
   post-trade long/short classification cannot be inferred. This accounting
   assertion currently turns 92 completed lifecycle attempts into
   `receipt_analysis_failed` across otherwise unrelated protocols.
2. Classify the result as `state_inference_failed` (or a dedicated position
   classification outcome), rather than `receipt_analysis_failed`, when
   transaction and manager analysis succeeded but portfolio inspection fails.
   Preserve the transaction evidence and decoded amounts in either case.
3. Do not run simulated Lagoon/ERC-7540 settlement through a signer-less
   provider. The 28 Lagoon rows return JSON-RPC `No Signer available`; the
   runner should classify this as infrastructure or explicitly unsupported
   settlement before attempting the request.
4. Treat cSigma's `maxRedeem` constraint as an amount-aware redemption-capacity
   result, not a receipt-analysis error after a successful deposit.
5. Keep the Accountable Monad satellite revert (`0x5945ea56`) as a structured
   transaction-reverted result with decoded custom-error context.

These are executor-side reporting and lifecycle issues. They obscure protocol
coverage and should be addressed before using this matrix as an adapter-success
metric.
