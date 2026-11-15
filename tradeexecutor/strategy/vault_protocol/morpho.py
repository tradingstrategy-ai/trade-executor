"""Reviewed public recovery for two Morpho Vault V2 exit-route failures.

Api3 dCOMP USDC (``1:0x36cfe1568461e499391ef0a555300f1ae2da2439``)
and Purinta USDC (``1:0xc92a37fd0250f4eecf092960a2f70a1334217528``)
price shares against the whole vault, but their configured liquidity route
withdraws from just one Morpho Blue market. On the Ethereum fork at block
26402309, a new 1,001 USDC deposit went into the slower-earning exit market.
After 24 hours the owner's aggregate claim exceeded that market's position:
Api3 by 0.132654 USDC (1.324883 bps), Purinta by 0.308119 USDC (3.076804 bps).
Blue's upward-rounded supply-share subtraction underflowed with ``Panic(0x11)``.
Other existing vault markets had assets and cash; this was not token precision
loss or a shortage of total market liquidity.

The reviewed public remedy is ``VaultV2.forceDeallocate`` from another funded
market in the same adapter, followed by redemption of the remaining shares.
These are separate, non-atomic transactions: freed assets can be consumed
before the finish, leaving the paid ticket pending. See ``docs/vault-protocol-overlays.md``.
The vault charges a penalty by burning the owner's shares and retaining the
corresponding assets. Both deployments charged 1 bp at the studied block.
Freeing 1,001 USDC charged 0.100100 USDC; the control then redeemed all remaining
shares for 1,001.150301 and 1,001.325571 USDC respectively. Those are diagnostic
control figures, using a fixed 1,001 USDC deallocation. The implemented manager
conservatively frees the gross current claim, slightly more than principal.
The recorded public Safe/Guard rerun, started at 18:41:53.919123 UTC on
15 November 2026, paid 0.100126 / 0.100143 USDC and returned
1,001.150259 / 1,001.325492 USDC. These outcomes depend on
mining times and the particular fork; the cap is a rate, not those fixed amounts.
Redistribution of retained penalty assets and intervening interest affect the final proceeds.

This rule permits that public execution step, capped at 1 bp. The adapter must
still verify the actual failure, route, market funding, live penalty and mined
share burns. The executor checkpoints the paid ticket and finishes immediately
through its existing resolver, before the live account check. Shares retained in custody
remain reserved for this exit; custody reconciliation subtracts only the paid
burn, while trading availability reserves the whole request. The cap is not a
fixed withdrawal fee and must not be applied again to net proceeds. Normal direct redemptions incur no recovery penalty.
There is no shorter simulated holding period, looser tolerance, injected payout
liquidity or curator impersonation. Oracle fault controls left the failure
unchanged: these redemptions made no collateral-oracle calls. Neither the 0.5%
trade slippage setting nor the 2.5% wallet-share reconciliation epsilon repairs
the contract's arithmetic underflow.

The market configured for exit was
``0xb323495f7e4148be5643a4ea4a8221eef163e4bccfdedc2a6f4696baacbc86cc``.
The demonstrated alternate markets were ``0x24852d8d7464402ddcd717415e009d42bf7427d6a8893487f83c75ee0f4a0ea6``
(Api3) and ``0x31a277fde40c1bd37dd00cb2167fe1d5831b450efecc63323679228a101e9979``
(Purinta). These observations explain the rule; they are not hardcoded routing
instructions. Live market selection and ABI-backed calls belong in eth-defi.

Source analysis and exact integer/receipt controls:
https://github.com/tradingstrategy-ai/vault-notebooks/pull/25#issuecomment-6896601999
and ``notebooks/vault-of-vaults-round-3/morpho-arithmetic.md`` in that repository.
Protocol implementation: https://github.com/morpho-org/vault-v2/blob/main/src/VaultV2.sol
(``exit``, ``forceDeallocate``); Blue's ``withdraw``:
https://github.com/morpho-org/morpho-blue/blob/main/src/Morpho.sol.

This is execution authorisation for these reviewed identities only. It does
not establish permissionless access, change strategy universe construction,
queue historical backtest redemptions or support the distinct VaultMorpho
protocol. Unreviewed vaults retain their existing behaviour.
"""

#: Maximum permitted configured recovery penalty in basis points (1 bp = 0.01%).
#: Keep names beside identities so a maintainer can locate the case study above.
REDEMPTION_RECOVERY_MAX_FEE_BPS: dict[tuple[int, str], int] = {
    (1, "0x36cfe1568461e499391ef0a555300f1ae2da2439"): 1,  # Api3 dCOMP USDC.
    (1, "0xc92a37fd0250f4eecf092960a2f70a1334217528"): 1,  # Purinta USDC.
}
