"""Mock-render regression tests for the xchain2 asset-weight legend."""

from base64 import b64decode, b64encode
from collections import Counter

import plotly.graph_objects as go
from tradingstrategy.chain import ChainId
from tradingstrategy.vault import VaultMetadata

from tradeexecutor.strategy.chart.asset_weight_legend import (
    ALLOCATION_X,
    ALLOCATION_ICON_WIDTH,
    CHAIN_X,
    CURATOR_X,
    HEADER_FONT_SIZE,
    IDENTITY_ICON_WIDTH,
    NAME_X,
    PROTOCOL_X,
    ROW_FONT_SIZE,
    AssetWeightLegendEntry,
    add_asset_weight_legend,
    allocation_swatch_data_url,
    merge_asset_weight_legend_entries,
)
from tradeexecutor.strategy.chart.vault_legend import get_vault_logo_coverage


#: Protocol counts in
#: ``14-backtest-phase-aware-hype-gains-lagoon-ipor-ember-yearn-40acres-csigma-yieldnest-plutus-25pct``.
XCHAIN2_PROTOCOL_COUNTS = {
    "40acres": 3,
    "csigma-finance": 2,
    "d2-finance": 1,
    "ember": 5,
    "gains-network": 2,
    "ipor-fusion": 17,
    "lagoon-finance": 36,
    "morpho": 1,
    "plutus": 2,
    "yearn": 51,
    "yieldnest": 1,
}

#: Exact protocol/curator combinations from the current 121-vault notebook
#: universe. This preserves the actual logo coverage rather than merely
#: exercising the same set of individual slugs.
XCHAIN2_VAULT_LOGO_COVERAGE = {
    ("40acres", None, False): 3,
    ("csigma-finance", None, False): 2,
    ("d2-finance", None, False): 1,
    ("ember", None, False): 4,
    ("ember", "apollo", False): 1,
    ("gains-network", "gains-network", True): 2,
    ("ipor-fusion", None, False): 4,
    ("ipor-fusion", "btcd-labs", False): 1,
    ("ipor-fusion", "clearstar-labs", False): 1,
    ("ipor-fusion", "ethena", False): 1,
    ("ipor-fusion", "harvest", False): 2,
    ("ipor-fusion", "hyperithm", False): 1,
    ("ipor-fusion", "ipor", False): 3,
    ("ipor-fusion", "tau", False): 4,
    ("lagoon-finance", None, False): 21,
    ("lagoon-finance", "722-capital", False): 1,
    ("lagoon-finance", "coinshift", False): 1,
    ("lagoon-finance", "der", False): 2,
    ("lagoon-finance", "detrade", False): 2,
    ("lagoon-finance", "gami", False): 2,
    ("lagoon-finance", "mt-pelerin", False): 2,
    ("lagoon-finance", "syntropia", False): 4,
    ("lagoon-finance", "tulipa-capital", False): 1,
    ("morpho", "steakhouse-financial", False): 1,
    ("plutus", None, False): 2,
    ("yearn", None, False): 43,
    ("yearn", "gauntlet", False): 1,
    ("yearn", "moonwell", False): 3,
    ("yearn", "steakhouse-financial", False): 1,
    ("yearn", "yearn", False): 3,
    ("yieldnest", "yieldnest", False): 1,
}


def _xchain2_mock_metadata() -> list[VaultMetadata]:
    """Create mock metadata with the notebook's complete logo coverage."""
    result = []
    for (protocol_slug, curator_slug, protocol_curator), count in XCHAIN2_VAULT_LOGO_COVERAGE.items():
        for _ in range(count):
            result.append(
                VaultMetadata(
                    vault_name=f"Mock vault {len(result):03}",
                    protocol_name=protocol_slug,
                    protocol_slug=protocol_slug,
                    features=[],
                    curator_slug=curator_slug,
                    protocol_curator=protocol_curator,
                ),
            )

    assert len(result) == 121
    return result


def _mock_chain_icon_data_url(chain_id: int) -> str:
    """Return a deterministic in-memory chain icon without network access."""
    colour = {
        ChainId.ethereum.value: "#627eea",
        ChainId.base.value: "#0052ff",
        ChainId.arbitrum.value: "#2d85d5",
        ChainId.avalanche.value: "#e84142",
    }[chain_id]
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><circle cx="16" cy="16" r="15" fill="{colour}"/></svg>'
    return f"data:image/svg+xml;base64,{b64encode(svg.encode()).decode('ascii')}"


def test_xchain2_logo_coverage_has_only_documented_unavailable_curator() -> None:
    """All current xchain2 protocol and identified-curator logos are covered.

    The mock data has the full 121-vault protocol/curator coverage of the
    notebook. DER is the sole exception because its own curator metadata says
    the operator has no public identity or logo source.
    """
    metadata_items = _xchain2_mock_metadata()
    assert Counter(
        (metadata.protocol_slug, metadata.curator_slug, metadata.protocol_curator)
        for metadata in metadata_items
    ) == XCHAIN2_VAULT_LOGO_COVERAGE

    coverage = get_vault_logo_coverage(metadata_items)
    protocol_rows = [row for row in coverage if row.kind == "protocol"]
    curator_rows = [row for row in coverage if row.kind == "curator"]

    assert Counter({row.slug: row.vault_count for row in protocol_rows}) == XCHAIN2_PROTOCOL_COUNTS
    assert not [row for row in protocol_rows if row.status != "available"]

    unavailable = [row for row in curator_rows if row.status == "not_available"]
    assert [(row.slug, row.vault_count) for row in unavailable] == [("der", 2)]
    assert unavailable[0].reason == "Anonymous curator; its metadata has no public identity, website, or logo source."
    assert not [row for row in curator_rows if row.status == "missing"]


def test_allocation_swatch_uses_trace_fill_and_plotly_replace_pattern() -> None:
    """Keep the allocation sample faithful to the Plotly trace fill encoding."""
    source = allocation_swatch_data_url(
        fill_colour="#00ff66",
        pattern_shape="x",
        pattern_size=5,
        pattern_solidity=0.8,
        pattern_fillmode="replace",
        pattern_foreground_colour=None,
        pattern_background_colour=None,
        pattern_foreground_opacity=None,
    )

    encoded_svg = source.removeprefix("data:image/svg+xml;base64,")
    svg = b64decode(encoded_svg).decode("utf-8")
    assert '<rect width="24" height="24" fill="#111111"/>' in svg
    assert 'stroke="#00ff66"' in svg
    assert "<pattern" in svg


def test_merged_cross_chain_vault_keeps_all_chain_and_protocol_icons(tmp_path) -> None:
    """gTrade's Base and Arbitrum allocations share one chart label."""
    gtrade_metadata = next(
        metadata
        for metadata in _xchain2_mock_metadata()
        if metadata.protocol_slug == "gains-network"
    )
    label = "gTrade (Gains Network USDC)"
    entries = merge_asset_weight_legend_entries(
        [
            AssetWeightLegendEntry(label, "#00ff66", ChainId.base.value, gtrade_metadata),
            AssetWeightLegendEntry(label, "#00ff66", ChainId.arbitrum.value, gtrade_metadata),
        ],
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry.chain_id is None
    assert entry.chain_ids == tuple(sorted((ChainId.base.value, ChainId.arbitrum.value)))
    assert entry.metadata is gtrade_metadata

    figure = go.Figure(
        go.Scatter(
            name=label,
            x=[0, 1],
            y=[1, 2],
            fill="tozeroy",
            fillcolor="#00ff66",
            fillpattern={"shape": "x", "size": 5, "solidity": 0.8},
        ),
    )
    figure.update_layout(template="plotly_dark")
    add_asset_weight_legend(figure, entries, chain_icon_resolver=_mock_chain_icon_data_url)

    gtrade_row_images = [image for image in figure.layout.images if image.y == 0.955]
    chain_images = [image for image in gtrade_row_images if CHAIN_X - 0.02 < image.x < CHAIN_X + 0.02]
    assert len(chain_images) == 2
    assert {image.x for image in chain_images} == {CHAIN_X - 0.0099, CHAIN_X + 0.0099}
    assert all(image.sizex == IDENTITY_ICON_WIDTH for image in chain_images)
    assert any(image.x == PROTOCOL_X for image in gtrade_row_images)
    assert not any(image.x == CURATOR_X for image in gtrade_row_images)

    rendered = tmp_path / "gtrade-asset-weight-legend.png"
    figure.write_image(rendered, width=1800, height=900)
    assert rendered.read_bytes().startswith(b"\x89PNG")


def test_render_xchain2_legend_has_aligned_columns(tmp_path) -> None:
    """Render all 121 mock vaults and keep each heading/icon column aligned."""
    metadata_items = _xchain2_mock_metadata()
    chain_ids = [
        ChainId.ethereum.value,
        ChainId.base.value,
        ChainId.arbitrum.value,
        ChainId.avalanche.value,
    ]
    traces = []
    entries = []
    for index, metadata in enumerate(metadata_items):
        colour = f"hsl({index * 17 % 360}, 65%, 55%)"
        traces.append(
            go.Scatter(
                name=metadata.vault_name,
                x=[0, 1],
                y=[index, index + 1],
                fill="tozeroy" if index == 0 else "tonexty",
                fillcolor=colour,
                fillpattern={
                    "shape": "-" if index == 0 else "x",
                    "size": 5,
                    "solidity": 0.8,
                },
                line={"color": "#000000"},
            ),
        )
        entries.append(
            AssetWeightLegendEntry(
                label=metadata.vault_name,
                colour=colour,
                chain_id=chain_ids[index % len(chain_ids)],
                metadata=metadata,
            ),
        )

    figure = go.Figure(traces)
    figure.update_layout(template="plotly_dark")
    add_asset_weight_legend(figure, entries, chain_icon_resolver=_mock_chain_icon_data_url)

    headers = figure.layout.annotations[:5]
    assert [(header.text, header.x, header.xanchor) for header in headers] == [
        ("A", ALLOCATION_X, "center"),
        ("C", CHAIN_X, "center"),
        ("P", PROTOCOL_X, "center"),
        ("C", CURATOR_X, "center"),
        ("Name", NAME_X, "left"),
    ]
    assert all(header.font.size == HEADER_FONT_SIZE for header in headers)
    assert all(annotation.font.size == ROW_FONT_SIZE for annotation in figure.layout.annotations[5:])
    assert not figure.layout.shapes
    allocation_images = [image for image in figure.layout.images if image.x == ALLOCATION_X]
    assert len(allocation_images) == len(metadata_items)
    assert all(image.sizex == ALLOCATION_ICON_WIDTH for image in allocation_images)
    assert all(image.xanchor == "center" and image.yanchor == "middle" for image in figure.layout.images)
    assert {image.x for image in figure.layout.images}.issubset({ALLOCATION_X, CHAIN_X, PROTOCOL_X, CURATOR_X})

    first_swatch_svg = b64decode(
        allocation_images[0].source.removeprefix("data:image/svg+xml;base64,"),
    ).decode("utf-8")
    assert 'stroke="hsl(0, 65%, 55%)"' in first_swatch_svg
    assert "#000000" not in first_swatch_svg
    assert "<pattern" in first_swatch_svg

    rendered = tmp_path / "xchain2-asset-weight-legend.png"
    figure.write_image(rendered, width=1800, height=figure.layout.height)
    assert rendered.read_bytes().startswith(b"\x89PNG")
