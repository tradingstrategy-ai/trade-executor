"""Aligned icon legends for asset-weight charts."""

import json
from base64 import b64encode
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from io import BytesIO
from math import sqrt
from pathlib import Path
from typing import Literal
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import tradingstrategy.chain as chain_module
from PIL import Image, ImageOps
from plotly.graph_objects import Figure
from tradingstrategy.chain import ChainId
from tradingstrategy.vault import VaultMetadata

from tradeexecutor.strategy.chart.vault_legend import resolve_vault_legend_logos


@dataclass(frozen=True, slots=True)
class AssetWeightLegendEntry:
    """Metadata displayed beside one asset-weight chart trace."""

    #: Label shared with the Plotly trace name.
    label: str

    #: Trace fill colour.
    colour: str

    #: Chain ID for the holding, if known.
    chain_id: int | None

    #: JSON-backed vault metadata, if the holding is a vault.
    metadata: VaultMetadata | None

    #: All chains represented by this trace. A same-named asset may be
    #: aggregated across chains by the asset-weight chart, but the static
    #: legend renders a separate row for every chain.
    chain_ids: tuple[int, ...] = ()

    #: Capital- and time-weighted annualised average yield, in percent.
    annualised_yield_percent: float = 0.0


#: Horizontal centres in Plotly paper coordinates. Header and row elements use
#: exactly the same values so column alignment does not depend on whitespace.
ALLOCATION_X = 0.580
CHAIN_X = 0.609
PROTOCOL_X = 0.635
CURATOR_X = 0.661
NAME_X = 0.682

ROW_TOP_Y = 0.960
ROW_BOTTOM_Y = 0.025
HEADER_Y = 0.985

ALLOCATION_ICON_WIDTH = 0.0293
IDENTITY_ICON_WIDTH = 0.024
MAX_ICON_HEIGHT = 0.032
HEADER_FONT_SIZE = 14
ROW_FONT_SIZE = 16

#: The below-chart legend keeps the original chart area and appends 40 pixels
#: of height for each row.
VERTICAL_FIGURE_WIDTH = 2200
VERTICAL_BASE_CHART_HEIGHT = 900
VERTICAL_ROW_HEIGHT_PX = 40
VERTICAL_HEADER_HEIGHT_PX = 40
VERTICAL_LEGEND_TOP_MARGIN_PX = 40
VERTICAL_LEGEND_BOTTOM_MARGIN_PX = 40
VERTICAL_PLOT_LEFT_MARGIN_PX = 80
VERTICAL_PLOT_RIGHT_MARGIN_PX = 8
VERTICAL_PLOT_TOP_MARGIN_PX = 80
VERTICAL_PLOT_WIDTH_PX = VERTICAL_FIGURE_WIDTH - VERTICAL_PLOT_LEFT_MARGIN_PX - VERTICAL_PLOT_RIGHT_MARGIN_PX
VERTICAL_PLOT_HEIGHT_PX = VERTICAL_BASE_CHART_HEIGHT - VERTICAL_PLOT_TOP_MARGIN_PX - VERTICAL_HEADER_HEIGHT_PX

#: Identity images use ``contain`` and need a square bounding box. Otherwise a
#: square logo is centred in a much wider invisible image box, creating the
#: appearance of excessive padding between legend columns.
VERTICAL_ALLOCATION_ICON_WIDTH = ALLOCATION_ICON_WIDTH * 2
VERTICAL_ICON_HEIGHT = VERTICAL_ROW_HEIGHT_PX / VERTICAL_PLOT_HEIGHT_PX
VERTICAL_IDENTITY_ICON_WIDTH = VERTICAL_ROW_HEIGHT_PX / VERTICAL_PLOT_WIDTH_PX
VERTICAL_ALLOCATION_ICON_WIDTH_PX = VERTICAL_ALLOCATION_ICON_WIDTH * VERTICAL_PLOT_WIDTH_PX
VERTICAL_COLUMN_GAP_PX = 12
VERTICAL_NAME_GAP_PX = 16
VERTICAL_LEGEND_LEFT_PX = 100

#: Horizontal centres for the below-chart legend, derived from the visible
#: icon bounds and pixel gaps rather than arbitrary paper-coordinate spacing.
VERTICAL_ALLOCATION_X = (
    VERTICAL_LEGEND_LEFT_PX
    + VERTICAL_ALLOCATION_ICON_WIDTH_PX / 2
    - VERTICAL_PLOT_LEFT_MARGIN_PX
) / VERTICAL_PLOT_WIDTH_PX
VERTICAL_CHAIN_X = (
    VERTICAL_LEGEND_LEFT_PX
    + VERTICAL_ALLOCATION_ICON_WIDTH_PX
    + VERTICAL_COLUMN_GAP_PX
    + VERTICAL_ROW_HEIGHT_PX / 2
    - VERTICAL_PLOT_LEFT_MARGIN_PX
) / VERTICAL_PLOT_WIDTH_PX
VERTICAL_PROTOCOL_X = VERTICAL_CHAIN_X + (
    VERTICAL_ROW_HEIGHT_PX + VERTICAL_COLUMN_GAP_PX
) / VERTICAL_PLOT_WIDTH_PX
VERTICAL_CURATOR_X = VERTICAL_PROTOCOL_X + (
    VERTICAL_ROW_HEIGHT_PX + VERTICAL_COLUMN_GAP_PX
) / VERTICAL_PLOT_WIDTH_PX
VERTICAL_NAME_X = VERTICAL_CURATOR_X + (
    VERTICAL_ROW_HEIGHT_PX / 2 + VERTICAL_NAME_GAP_PX
) / VERTICAL_PLOT_WIDTH_PX

VERTICAL_HEADER_FONT_SIZE = HEADER_FONT_SIZE * 2
VERTICAL_ROW_FONT_SIZE = ROW_FONT_SIZE * 2


def _entry_chain_ids(entry: AssetWeightLegendEntry) -> tuple[int, ...]:
    """Return the distinct chains represented by a legend entry."""
    chain_ids = entry.chain_ids or ((entry.chain_id,) if entry.chain_id else ())
    return tuple(dict.fromkeys(chain_ids))


def _has_same_logo_identity(left: VaultMetadata, right: VaultMetadata) -> bool:
    """Check whether two aggregated vaults can safely share one logo set."""
    return (
        left.protocol_slug,
        left.curator_slug,
        left.protocol_curator,
    ) == (
        right.protocol_slug,
        right.curator_slug,
        right.protocol_curator,
    )


def merge_asset_weight_legend_entries(
    entries: Iterable[AssetWeightLegendEntry],
) -> list[AssetWeightLegendEntry]:
    """Merge entries whose asset-weight trace shares the same label and chain.

    Asset weights are grouped by their visible label, which means one chart
    trace can aggregate the same vault on several chains. Keep those chains
    as separate legend rows: each row has one allocation swatch, chain icon,
    protocol logo, curator logo, and vault name. If duplicate positions on a
    single chain disagree about their vault identity, omit the protocol and
    curator logos rather than associating a misleading identity with the row.
    """
    merged: dict[tuple[str, int | None], AssetWeightLegendEntry] = {}
    for entry in entries:
        chain_ids = _entry_chain_ids(entry) or (None,)
        for chain_id in chain_ids:
            key = (entry.label, chain_id)
            previous = merged.get(key)
            if previous is None:
                merged[key] = AssetWeightLegendEntry(
                    label=entry.label,
                    colour=entry.colour,
                chain_id=chain_id,
                chain_ids=(chain_id,) if chain_id else (),
                metadata=entry.metadata,
                annualised_yield_percent=entry.annualised_yield_percent,
            )
                continue

            metadata = (
                previous.metadata
                if previous.metadata and entry.metadata and _has_same_logo_identity(previous.metadata, entry.metadata)
                else None
            )
            merged[key] = AssetWeightLegendEntry(
                label=previous.label,
                colour=previous.colour,
                chain_id=chain_id,
                chain_ids=(chain_id,) if chain_id else (),
                metadata=metadata,
                annualised_yield_percent=previous.annualised_yield_percent,
            )

    return list(merged.values())


def _plotly_fill_pattern_svg_path(shape: str, size: float, solidity: float) -> tuple[float, str, float] | None:
    """Translate a Plotly scatter fill-pattern shape to an SVG tile.

    The asset-weight chart currently uses ``"-"`` for reserves and ``"x"``
    for vaults. The equations mirror Plotly's own SVG tiles so the allocation
    swatch communicates the same visual encoding as the chart area.

    :param shape:
        Plotly scatter ``fillpattern.shape`` value.
    :param size:
        Plotly scatter ``fillpattern.size`` value in pixels.
    :param solidity:
        Plotly scatter ``fillpattern.solidity`` value from zero to one.
    :return:
        SVG tile size, SVG path data, and stroke width, or ``None`` for a
        solid fill.
    """
    if not shape:
        return None

    if shape == "-":
        return size, f"M0,{size / 2}L{size},{size / 2}", size * solidity
    if shape == "|":
        return size, f"M{size / 2},0L{size / 2},{size}", size * solidity
    if shape == "+":
        return size, f"M0,{size / 2}L{size},{size / 2}M{size / 2},0L{size / 2},{size}", size * solidity

    diagonal_size = size * sqrt(2)
    diagonal_stroke_width = size * (1 - sqrt(1 - solidity))
    if shape == "/":
        return diagonal_size, f"M{-diagonal_size / 4},{diagonal_size / 4}l{diagonal_size / 2},{-diagonal_size / 2}M0,{diagonal_size}L{diagonal_size},0M{3 * diagonal_size / 4},{5 * diagonal_size / 4}l{diagonal_size / 2},{-diagonal_size / 2}", diagonal_stroke_width
    if shape == "\\":
        return diagonal_size, f"M0,0L{diagonal_size},{diagonal_size}M{-diagonal_size / 4},{3 * diagonal_size / 4}l{diagonal_size / 2},{diagonal_size / 2}M{3 * diagonal_size / 4},{-diagonal_size / 4}l{diagonal_size / 2},{diagonal_size / 2}", diagonal_stroke_width
    if shape == "x":
        return diagonal_size, f"M{-diagonal_size / 4},{diagonal_size / 4}l{diagonal_size / 2},{-diagonal_size / 2}M0,{diagonal_size}L{diagonal_size},0M{3 * diagonal_size / 4},{5 * diagonal_size / 4}l{diagonal_size / 2},{-diagonal_size / 2}M{3 * diagonal_size / 4},{-diagonal_size / 4}l{diagonal_size / 2},{diagonal_size / 2}M0,0L{diagonal_size},{diagonal_size}M{-diagonal_size / 4},{3 * diagonal_size / 4}l{diagonal_size / 2},{diagonal_size / 2}", diagonal_stroke_width

    return None


@lru_cache(maxsize=None)
def allocation_swatch_data_url(
    fill_colour: str,
    pattern_shape: str,
    pattern_size: float,
    pattern_solidity: float,
    pattern_fillmode: str,
    pattern_foreground_colour: str | None,
    pattern_background_colour: str | None,
    pattern_foreground_opacity: float | None,
) -> str:
    """Create an SVG allocation swatch matching a Plotly scatter trace.

    Pattern fills are not supported on Plotly layout shapes. Embedding an SVG
    tile lets the static legend mirror the trace's fill colour and pattern,
    including the transparent gaps used by Plotly's default ``replace`` mode.

    :param fill_colour:
        Trace ``fillcolor``.
    :param pattern_shape:
        Trace ``fillpattern.shape``.
    :param pattern_size:
        Trace ``fillpattern.size``.
    :param pattern_solidity:
        Trace ``fillpattern.solidity``.
    :param pattern_fillmode:
        Trace ``fillpattern.fillmode``.
    :param pattern_foreground_colour:
        Explicit foreground colour, if defined on the trace.
    :param pattern_background_colour:
        Explicit background colour, if defined on the trace.
    :param pattern_foreground_opacity:
        Explicit foreground opacity, if defined on the trace.
    :return:
        SVG data URL suitable for :meth:`plotly.graph_objects.Figure.add_layout_image`.
    """
    tile = _plotly_fill_pattern_svg_path(pattern_shape, pattern_size, pattern_solidity)
    if tile is None:
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><rect width="24" height="24" fill="{escape(fill_colour)}"/></svg>'
    else:
        tile_size, path, stroke_width = tile
        fillmode = pattern_fillmode or "replace"
        foreground_colour = pattern_foreground_colour or fill_colour
        foreground_opacity = pattern_foreground_opacity
        if foreground_opacity is None:
            foreground_opacity = 0.5 if fillmode == "overlay" else 1
        background_colour = pattern_background_colour
        if background_colour is None:
            background_colour = fill_colour if fillmode == "overlay" else "#111111"
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            '<defs>'
            f'<pattern id="pattern" width="{tile_size}" height="{tile_size}" patternUnits="userSpaceOnUse">'
            f'<path d="{path}" fill="none" stroke="{escape(foreground_colour)}" '
            f'stroke-width="{stroke_width}" opacity="{foreground_opacity}"/>'
            '</pattern>'
            '</defs>'
            f'<rect width="24" height="24" fill="{escape(background_colour)}"/>'
            '<rect width="24" height="24" fill="url(#pattern)"/>'
            '</svg>'
        )
    encoded = b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _normalise_dark_chart_icon(image: Image.Image) -> Image.Image:
    """Centre and normalise a transparent icon for a dark chart background."""
    image = image.convert("RGBA")
    alpha = image.getchannel("A")
    bounding_box = alpha.getbbox()
    if bounding_box is None:
        return image

    image = image.crop(bounding_box)
    opaque_pixels = [pixel for pixel in image.get_flattened_data() if pixel[3] > 16]
    mean_luminance = sum((pixel[0] + pixel[1] + pixel[2]) / 3 for pixel in opaque_pixels) / len(opaque_pixels)
    if mean_luminance < 70:
        red, green, blue, alpha = image.split()
        inverted_rgb = ImageOps.invert(Image.merge("RGB", (red, green, blue)))
        image = Image.merge("RGBA", (*inverted_rgb.split(), alpha))

    target_size = 256
    content_size = round(target_size * 0.82)
    image.thumbnail((content_size, content_size), Image.Resampling.LANCZOS)
    normalised = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    offset = ((target_size - image.width) // 2, (target_size - image.height) // 2)
    normalised.alpha_composite(image, dest=offset)
    return normalised


@lru_cache(maxsize=None)
def local_png_data_url(path: Path) -> str:
    """Embed and normalise a local PNG icon for a dark static chart."""
    normalised = _normalise_dark_chart_icon(Image.open(path))
    output = BytesIO()
    normalised.save(output, format="PNG", optimise=True)
    encoded = b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@lru_cache(maxsize=None)
def resolve_chain_icon_data_url(chain_id: int) -> str | None:
    """Resolve a ChainId asset into a static-chart data URL.

    Chainlist's bundled icon asset is preferred. The ChainId SVG URL is used
    only for chains such as Arbitrum that have no bundled icon record.
    """
    chain = ChainId(chain_id)
    icon_name = chain.data.get("icon")
    if icon_name:
        icons_data_dir = Path(chain_module.__file__).parent / "chains" / "_data"
        icon_metadata_path = icons_data_dir / "icons" / f"{icon_name}.json"
        try:
            icon_metadata = json.loads(icon_metadata_path.read_text())[0]
            content_id = icon_metadata["url"].removeprefix("ipfs://")
            icon_path = icons_data_dir / "iconsDownload" / content_id
            if icon_path.is_file():
                return local_png_data_url(icon_path)
        except (IndexError, KeyError, OSError, ValueError):
            pass

    try:
        request = Request(chain.get_svg_icon_link(), headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=10) as response:
            encoded = b64encode(response.read()).decode("ascii")
        return f"data:image/svg+xml;base64,{encoded}"
    except (KeyError, OSError, ValueError):
        return None


def calculate_capital_time_allocation_percentages(figure: Figure) -> dict[str, float]:
    """Calculate each trace's share of the portfolio's capital-time allocation.

    The asset-weight chart holds USD values, not normalised percentages. This
    integrates each trace over the chart's timestamps and divides it by the
    integrated total portfolio value. Unlike a simple average of sample
    percentages, irregular time intervals contribute in proportion to their
    duration.

    Missing trace values represent no allocation and are treated as zero.
    A single-sample chart has no time interval, so its instantaneous asset
    weights are used as the best available allocation proxy.
    """
    trace_values = [
        (trace.name, np.nan_to_num(np.asarray(trace.y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0))
        for trace in figure.data
        if trace.name and trace.y is not None
    ]
    if not trace_values:
        return {}

    sample_count = len(trace_values[0][1])
    trace_values = [item for item in trace_values if len(item[1]) == sample_count]
    if not trace_values or sample_count == 0:
        return {}

    values = np.vstack([value for _name, value in trace_values])
    if sample_count == 1:
        capital_times = values[:, 0]
    else:
        x_values = np.asarray(figure.data[0].x)
        if len(x_values) != sample_count:
            durations = np.ones(sample_count - 1)
        elif np.issubdtype(x_values.dtype, np.number):
            durations = np.diff(x_values.astype(float))
        elif np.issubdtype(x_values.dtype, np.datetime64):
            durations = np.diff(x_values.astype("datetime64[ns]").astype("int64")) / 1_000_000_000
        else:
            timestamps = pd.to_datetime(x_values)
            durations = np.diff(timestamps.asi8) / 1_000_000_000

        durations = np.maximum(durations, 0)
        capital_times = ((values[:, :-1] + values[:, 1:]) / 2 * durations).sum(axis=1)

    total_capital_time = capital_times.sum()
    if total_capital_time <= 0:
        return {}

    return {
        name: float(capital_time / total_capital_time * 100)
        for (name, _value), capital_time in zip(trace_values, capital_times, strict=True)
    }


def add_asset_weight_legend(
    figure: Figure,
    entries: Iterable[AssetWeightLegendEntry],
    *,
    chain_icon_resolver: Callable[[int], str | None] = resolve_chain_icon_data_url,
    legend_layout: Literal["horizontal", "vertical"] = "horizontal",
) -> None:
    """Replace a native legend with aligned allocation and identity columns.

    The Plotly legend cannot render images. This layout puts all column headers
    and icon boxes on fixed paper-coordinate centres, so absent curator logos
    reserve an empty but aligned slot. ``"horizontal"`` keeps the compact
    right-side legend. ``"vertical"`` places doubled-size rows below the chart,
    adds 40 pixels per row, and leaves 40-pixel top and bottom legend margins.

    :param figure:
        Asset-weight figure whose trace names identify legend entries.
    :param entries:
        Trace metadata. Same-named entries on different chains become
        separate legend rows.
    :param chain_icon_resolver:
        Injectable resolver used by tests to avoid external icon downloads.
    :param legend_layout:
        ``"horizontal"`` for the compact right-side legend or ``"vertical"``
        for the enlarged below-chart legend.
    """
    if legend_layout not in {"horizontal", "vertical"}:
        raise ValueError(f"Unsupported asset-weight legend layout: {legend_layout}")

    merged_entries = merge_asset_weight_legend_entries(entries)
    capital_time_allocations = calculate_capital_time_allocation_percentages(figure)
    entries_by_label: dict[str, list[AssetWeightLegendEntry]] = {}
    for entry in merged_entries:
        entries_by_label.setdefault(entry.label, []).append(entry)

    legend_rows = [
        (trace, entry)
        for trace in figure.data
        for entry in entries_by_label.get(trace.name, [None])
    ]
    row_count = max(len(legend_rows), 1)
    if legend_layout == "horizontal":
        row_step = (ROW_TOP_Y - ROW_BOTTOM_Y) / max(row_count - 1, 1)
        icon_height = min(MAX_ICON_HEIGHT, row_step * 0.93)
        figure_height = max(900, (row_count + 1) * 32)
        allocation_x = ALLOCATION_X
        chain_x = CHAIN_X
        protocol_x = PROTOCOL_X
        curator_x = CURATOR_X
        name_x = NAME_X
        header_y = HEADER_Y
        header_font_size = HEADER_FONT_SIZE
        row_font_size = ROW_FONT_SIZE
        allocation_icon_width = ALLOCATION_ICON_WIDTH
        identity_icon_width = IDENTITY_ICON_WIDTH

        figure.update_layout(
            showlegend=False,
            width=1800,
            height=figure_height,
            margin={"r": 8},
        )
        figure.update_xaxes(domain=[0, 0.560])

        def get_row_y(index: int) -> float:
            return ROW_TOP_Y - index * row_step

    else:
        figure_height = (
            VERTICAL_BASE_CHART_HEIGHT
            + row_count * VERTICAL_ROW_HEIGHT_PX
            + VERTICAL_LEGEND_TOP_MARGIN_PX
            + VERTICAL_LEGEND_BOTTOM_MARGIN_PX
        )
        icon_height = VERTICAL_ICON_HEIGHT
        allocation_x = VERTICAL_ALLOCATION_X
        chain_x = VERTICAL_CHAIN_X
        protocol_x = VERTICAL_PROTOCOL_X
        curator_x = VERTICAL_CURATOR_X
        name_x = VERTICAL_NAME_X
        header_y = -(
            VERTICAL_LEGEND_TOP_MARGIN_PX
            + VERTICAL_HEADER_HEIGHT_PX / 2
        ) / VERTICAL_PLOT_HEIGHT_PX
        header_font_size = VERTICAL_HEADER_FONT_SIZE
        row_font_size = VERTICAL_ROW_FONT_SIZE
        allocation_icon_width = VERTICAL_ALLOCATION_ICON_WIDTH
        identity_icon_width = VERTICAL_IDENTITY_ICON_WIDTH

        figure.update_layout(
            showlegend=False,
            width=VERTICAL_FIGURE_WIDTH,
            height=figure_height,
            margin={
                "l": VERTICAL_PLOT_LEFT_MARGIN_PX,
                "r": VERTICAL_PLOT_RIGHT_MARGIN_PX,
                "t": VERTICAL_PLOT_TOP_MARGIN_PX,
                "b": (
                    VERTICAL_LEGEND_TOP_MARGIN_PX
                    + VERTICAL_HEADER_HEIGHT_PX
                    + row_count * VERTICAL_ROW_HEIGHT_PX
                    + VERTICAL_LEGEND_BOTTOM_MARGIN_PX
                ),
            },
        )
        figure.update_xaxes(domain=[0, 1])

        def get_row_y(index: int) -> float:
            return -(
                VERTICAL_LEGEND_TOP_MARGIN_PX
                +
                VERTICAL_HEADER_HEIGHT_PX
                + (index + 0.5) * VERTICAL_ROW_HEIGHT_PX
            ) / VERTICAL_PLOT_HEIGHT_PX

    for label, x, xanchor in (
        ("A", allocation_x, "center"),
        ("C", chain_x, "center"),
        ("P", protocol_x, "center"),
        ("C", curator_x, "center"),
        ("Name", name_x, "left"),
    ):
        figure.add_annotation(
            x=x,
            y=header_y,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            xanchor=xanchor,
            yanchor="middle",
            font={"size": header_font_size, "color": "#a8b1c1"},
        )

    for index, (trace, entry) in enumerate(legend_rows):
        y = get_row_y(index)
        colour = trace.fillcolor or trace.line.color or (entry.colour if entry else "#a8b1c1")
        fillpattern = trace.fillpattern
        allocation_source = allocation_swatch_data_url(
            fill_colour=colour,
            pattern_shape=fillpattern.shape or "",
            pattern_size=fillpattern.size or 5,
            pattern_solidity=fillpattern.solidity if fillpattern.solidity is not None else 0.5,
            pattern_fillmode=fillpattern.fillmode or "replace",
            pattern_foreground_colour=fillpattern.fgcolor,
            pattern_background_colour=fillpattern.bgcolor,
            pattern_foreground_opacity=fillpattern.fgopacity,
        )
        figure.add_layout_image(
            source=allocation_source,
            x=allocation_x,
            y=y,
            xref="paper",
            yref="paper",
            sizex=allocation_icon_width,
            sizey=icon_height,
            xanchor="center",
            yanchor="middle",
            sizing="stretch",
            layer="above",
        )

        chain_id = entry.chain_id if entry else None
        chain_source = chain_icon_resolver(chain_id) if chain_id else None
        protocol_source = None
        curator_source = None
        if entry and entry.metadata:
            logos = resolve_vault_legend_logos(entry.metadata)
            protocol_source = local_png_data_url(logos.protocol) if logos.protocol else None
            curator_source = local_png_data_url(logos.curator) if logos.curator else None

        icon_specs = [
            (chain_x, chain_source, identity_icon_width),
            (protocol_x, protocol_source, identity_icon_width),
            (curator_x, curator_source, identity_icon_width),
        ]

        for x, source, icon_width in icon_specs:
            if source:
                figure.add_layout_image(
                    source=source,
                    x=x,
                    y=y,
                    xref="paper",
                    yref="paper",
                    sizex=icon_width,
                    sizey=icon_height,
                    xanchor="center",
                    yanchor="middle",
                    sizing="contain",
                    layer="above",
                )

        figure.add_annotation(
            x=name_x,
            y=y,
            xref="paper",
            yref="paper",
            text=(
                (
                    f'{trace.name}: <span style="color:#aaa">'
                    f'allocated <span style="color:#ffd700">{capital_time_allocations[trace.name]:.1f}%</span>, '
                    f'yield <span style="color:#00ff66">{entry.annualised_yield_percent if entry else 0.0:.1f}%</span></span>'
                )
                if legend_layout == "vertical" and trace.name in capital_time_allocations
                else (
                    f'{trace.name}: <span style="color:#aaa">{capital_time_allocations[trace.name]:.1f}%</span>'
                    if trace.name in capital_time_allocations
                    else trace.name
                )
            ),
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font={"size": row_font_size, "color": "#e5ecf6"},
        )
