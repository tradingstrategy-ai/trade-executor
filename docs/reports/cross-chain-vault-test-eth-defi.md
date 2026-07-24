# Eth-defi cross-chain vault test report

## Run

The complete 129-vault cross-chain simulated matrix was run on 2026-07-24
against `web3-ethereum-defi` master `b5803bdc5`.

The raw terminal classification is dominated by a trade-executor post-trade
state-inference failure. It is therefore not valid to count its 92
`receipt_analysis_failed` rows as eth-defi decoder failures. The actionable
eth-defi findings are below.

| Protocol | Vaults | Required eth-defi work |
|---|---:|---|
| Upshift | 1 | Implement a dedicated multi-asset deposit manager for Sentora USD Earn, including accepted-asset conversion, request construction and event analysis. |
| cSigma Finance | 2 | Expose owner- and amount-aware `maxRedeem` capacity through the manager and return a structured unavailable/capacity outcome before redemption construction. |
| D2 Finance | 1 | Make closed funding windows and zero `previewDeposit()` pricing a structured preflight result, including the next-open time when available. |
| Accountable | 1 | Decode custom error `0x5945ea56` and publish the caller/admission condition for the Monad satellite vault. |
| Lagoon/ERC-7540 | 28 | Provide a protocol-specific Anvil request/settle/claim simulation path, or explicitly advertise that it is unsupported without invoking a signer-less settlement call. |

## Evidence to recheck after executor fixes

Ember (5), IPOR Fusion (17), Morpho (11), Yearn (43), Euler (4), 40acres
(3), Gains Network (2), Plutus (2) and YieldNest (1) are currently masked by
the executor's post-trade state-inference failure. Re-run those rows after the
trade-executor fixes above before changing their adapters or decoders.
