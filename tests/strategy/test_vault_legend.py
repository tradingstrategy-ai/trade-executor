"""Tests for local vault legend logo resolution."""

from tradingstrategy.vault import VaultMetadata

from tradeexecutor.strategy.chart.vault_legend import resolve_vault_legend_logos


def _metadata(**kwargs) -> VaultMetadata:
    """Create minimal vault metadata with optional curator overrides."""
    values = {
        "vault_name": "Morpho Gauntlet USDC",
        "protocol_name": "Morpho",
        "protocol_slug": "morpho",
        "features": [],
    }
    values.update(kwargs)
    return VaultMetadata(**values)


def test_resolve_vault_legend_logos_prefers_light_dark_chart_variants() -> None:
    """A dark chart uses local light protocol and curator logo variants."""
    logos = resolve_vault_legend_logos(_metadata(curator_slug="gauntlet"))

    assert logos.protocol.name == "light.png"
    assert logos.protocol.parent.name == "morpho"
    assert logos.curator.name == "light.png"
    assert logos.curator.parent.name == "gauntlet"


def test_resolve_vault_legend_logos_leaves_missing_curator_slot_empty() -> None:
    """Absent curator data or a missing local asset leaves the slot empty."""
    logos = resolve_vault_legend_logos(_metadata(curator_slug="unknown-curator"))

    assert logos.protocol is not None
    assert logos.curator is None
    assert resolve_vault_legend_logos(_metadata()).curator is None


def test_resolve_vault_legend_logos_deduplicates_protocol_curator() -> None:
    """A protocol curator is not shown a second time in the curator slot."""
    logos = resolve_vault_legend_logos(
        _metadata(curator_slug="morpho", curator_name="Morpho", protocol_curator=True),
    )

    assert logos.protocol is not None
    assert logos.curator is None
