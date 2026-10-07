# HyperCore permission clocks

HyperCore price timestamps, permission receipts and publication times are
separate clocks. The Trading Strategy client selects whole permission snapshots
at their first decision bucket, rounding availability upwards. It retains
`permission_observed_at`, `permission_provenance`, `permission_observation_id`
and source order. Repeated price projections do not refresh these clocks.

Recovered legacy flags without an independent receipt use the original price
key with `legacy_price_timestamp` provenance. This is the approved recovery
approximation, not evidence of the precise venue transition. Flags tagged
`corrupted_unknown` or `legacy_unverified` remain Unknown. Untagged corrupt
history cannot be identified automatically and must first be repaired or tagged.
Genuine responses, including successful
responses with missing flags, supersede inferred evidence. Conflicting genuine
snapshots at one receipt time resolve to Unknown. Retrospective uncertainty
intervals use their historical bounds, irrespective of migration publication.
Where no genuine response exists, a newer deny-only archive bound overrides
an older inferred flag. It can establish Closed from `evidence_available_at`,
never Open, and retains the absent permission receipt clock. The executor ages
this denial against its archive availability bound.

Backtest pricing checks freshness against the original, unrounded clock using
its data-delay tolerance (two days by default). Capacity has its own
`capacity_observed_at`; an inferred permission does not authenticate a carried
leader share or capacity cap. HyperCore Unknown blocks new deposits from the
existing 11 April 2026 cutoff onwards. Before that boundary the established
assumed-open policy remains in effect. Other chains keep their existing
historical availability policy.

## Independent history

The client can explicitly download `VaultDataset.hypercore_vault_permissions`
through the licence-authenticated `hypercore-vault-permissions` dataset route.
That route must be deployed before using it. The sidecar contains exact
observations, nullable flags and uncertainty intervals, independently of price
history. `read_vault_permission_history_parquet()` filters vault identities,
retaining earlier receipts rather than applying price-window predicates.

Pass a local sidecar matching the price generation to
`load_partial_data(vault_permission_history_path=...)`. Its state is converted
separately from prices and included in the indicator cache fingerprint.
Observations after the latest price can change availability without creating
price or TVL candles. Website loading uses the current licence-authenticated
`VaultDataClient`, including any verified price snapshot passed by the live
trigger. Removed public downloader APIs are not used.

Manifest v1 authenticates prices only. Manifest v2 stays rejected and no
unversioned sidecar is downloaded automatically during live loading. Enabling
v2 needs coordinated authenticated endpoint, client and executor support for
the same immutable price and permission generation. This change does not
publish, deploy or migrate production data.

## Recovery validation

The [2026 DOEZOE report](reports/doezoe-deposit-status-2026-after-client-fix.md)
checks the repaired projection and explicit sidecar through the actual client
converter and executor deposit APIs. Midnight status is Open through
21 September and Closed from 22 September through the snapshot's 6 October
cutoff. All 279 evaluated daily admission decisions agree with the archived
producer reference. On 3 April, an intervening corrupted-data boundary makes
the updated state Unknown instead of carrying inferred Open forwards; admission
still follows the pre-cutoff assumed-open policy. Later calendar dates are not
evaluated against this retained snapshot.

The report does not establish independently measured historical permissions:
DOEZOE has zero genuine permission receipts in these recovery files. It also
does not verify the current production deployment. The available earlier R2
archive is longer than the quoted partial archive and has 18 rows rather than
five on 28 September. Source hashes, precise scanner and Parquet clocks, and
the reproducible command are saved alongside the full daily table.

See [Trading Strategy issue 252](https://github.com/tradingstrategy-ai/trading-strategy/issues/252)
for the recovery policy and regression contract.
