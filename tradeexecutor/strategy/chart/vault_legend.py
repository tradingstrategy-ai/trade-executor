"""Local logo resolution for vault chart legends."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from eth_defi.vault.protocol_metadata import FORMATTED_LOGOS_DIR
from tradingstrategy.vault import VaultMetadata


#: Curators whose own metadata establishes that a public logo does not exist.
#:
#: Keep these separate from an accidental missing local asset so the chart
#: coverage audit can highlight new logo work without asking callers to infer
#: curator identities from vault names.
CURATOR_LOGO_UNAVAILABILITY_REASONS: dict[str, str] = {
    "der": "Anonymous curator; its metadata has no public identity, website, or logo source.",
}


@dataclass(frozen=True, slots=True)
class VaultLegendLogos:
    """Local protocol and curator logo paths for one vault legend entry."""

    #: Protocol logo path, if a formatted logo exists.
    protocol: Path | None

    #: Curator logo path, if the JSON identifies a distinct curator with a logo.
    curator: Path | None


@dataclass(frozen=True, slots=True)
class VaultLogoCoverage:
    """One grouped protocol or curator result from a vault-logo audit."""

    #: ``"protocol"`` or ``"curator"``.
    kind: str

    #: Protocol or curator slug. ``None`` when JSON did not supply a curator.
    slug: str | None

    #: Number of vault records represented by this result.
    vault_count: int

    #: ``"available"``, ``"missing"``, ``"not_available"`` or ``"omitted"``.
    status: str

    #: Human-readable explanation for the result.
    reason: str

    #: Resolved logo path when ``status`` is ``"available"``.
    path: Path | None


def _resolve_local_logo(slug: str | None, variants: tuple[str, ...]) -> Path | None:
    """Resolve the first available local formatted logo for a slug."""
    if not slug:
        return None

    logo_dir = FORMATTED_LOGOS_DIR / slug
    for variant in variants:
        candidate = logo_dir / f"{variant}.png"
        if candidate.is_file():
            return candidate

    return None


def resolve_vault_legend_logos(
    metadata: VaultMetadata,
    *,
    dark_background: bool = True,
) -> VaultLegendLogos:
    """Resolve locally stored protocol and curator logos for a vault.

    Curator identity is used only as supplied by the vault JSON through
    :class:`~tradingstrategy.vault.VaultMetadata`; this helper deliberately
    does not infer or re-identify a curator from the vault name.  On a dark
    chart, a light logo is preferred, followed by a generic and dark variant.

    :param metadata:
        Vault metadata loaded from ``top_vaults_by_chain.json``.
    :param dark_background:
        Select the suitable logo variant order for the chart background.
    :return:
        Resolved local protocol and curator logo paths. The curator path is
        ``None`` for an absent or protocol curator, and for missing assets.
    """
    variants = ("light", "generic", "dark") if dark_background else ("dark", "generic", "light")
    protocol = _resolve_local_logo(metadata.protocol_slug, variants)

    if metadata.protocol_curator or metadata.curator_slug == metadata.protocol_slug:
        curator = None
    else:
        curator = _resolve_local_logo(metadata.curator_slug, variants)

    return VaultLegendLogos(protocol=protocol, curator=curator)


def get_vault_logo_coverage(metadata_items: Iterable[VaultMetadata]) -> list[VaultLogoCoverage]:
    """Audit local protocol and curator logo coverage for vault metadata.

    The audit preserves the authoritative curator state from vault JSON. An
    absent curator slug and a protocol curator are reported as intentionally
    omitted; they are not treated as missing artwork.

    :param metadata_items:
        JSON-backed vault metadata records to audit.
    :return:
        Grouped protocol and curator logo coverage results.
    """
    coverage: Counter[tuple[str, str | None, str, str, Path | None]] = Counter()

    for metadata in metadata_items:
        logos = resolve_vault_legend_logos(metadata)
        protocol_status = "available" if logos.protocol else "missing"
        protocol_reason = "Local formatted logo found." if logos.protocol else "No local formatted logo asset."
        coverage[("protocol", metadata.protocol_slug, protocol_status, protocol_reason, logos.protocol)] += 1

        if metadata.curator_slug is None:
            curator_status = "omitted"
            curator_reason = "No curator_slug in authoritative vault JSON."
            curator_path = None
        elif metadata.protocol_curator or metadata.curator_slug == metadata.protocol_slug:
            curator_status = "omitted"
            curator_reason = "Curator is the protocol and is not repeated."
            curator_path = None
        elif logos.curator:
            curator_status = "available"
            curator_reason = "Local formatted logo found."
            curator_path = logos.curator
        elif metadata.curator_slug in CURATOR_LOGO_UNAVAILABILITY_REASONS:
            curator_status = "not_available"
            curator_reason = CURATOR_LOGO_UNAVAILABILITY_REASONS[metadata.curator_slug]
            curator_path = None
        else:
            curator_status = "missing"
            curator_reason = "No local formatted logo asset."
            curator_path = None

        coverage[("curator", metadata.curator_slug, curator_status, curator_reason, curator_path)] += 1

    return [
        VaultLogoCoverage(
            kind=kind,
            slug=slug,
            vault_count=vault_count,
            status=status,
            reason=reason,
            path=path,
        )
        for (kind, slug, status, reason, path), vault_count in sorted(
            coverage.items(),
            key=lambda item: (item[0][0], item[0][1] or "", item[0][2]),
        )
    ]
