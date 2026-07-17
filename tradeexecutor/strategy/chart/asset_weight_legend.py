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
from urllib.request import Request, urlopen

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
    #: aggregated across chains by the asset-weight chart.
    chain_ids: tuple[int, ...] = ()


#: Horizontal centres in Plotly paper coordinates. Header and row elements use
#: exactly the same values so column alignment does not depend on whitespace.
ALLOCATION_X = 0.628
CHAIN_X = 0.660
PROTOCOL_X = 0.693
CURATOR_X = 0.726
NAME_X = 0.753

ROW_TOP_Y = 0.955
ROW_BOTTOM_Y = 0.035
HEADER_Y = 0.982

ALLOCATION_ICON_WIDTH = 0.022
IDENTITY_ICON_WIDTH = 0.018
MAX_ICON_HEIGHT = 0.024
HEADER_FONT_SIZE = 12
ROW_FONT_SIZE = 13


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
    """Merge entries whose asset-weight trace shares the same label.

    Asset weights are grouped by their visible label. If a label aggregates
    the same vault identity on more than one chain, retain its protocol and
    curator metadata and render every represented chain icon. If identities
    differ, omit the logo instead of associating a misleading protocol with
    the aggregate.
    """
    merged: dict[str, AssetWeightLegendEntry] = {}
    for entry in entries:
        previous = merged.get(entry.label)
        if previous is None:
            merged[entry.label] = entry
            continue

        chain_ids = tuple(sorted(set(_entry_chain_ids(previous)) | set(_entry_chain_ids(entry))))
        metadata = (
            previous.metadata
            if previous.metadata and entry.metadata and _has_same_logo_identity(previous.metadata, entry.metadata)
            else None
        )
        merged[entry.label] = AssetWeightLegendEntry(
            label=previous.label,
            colour=previous.colour,
            chain_id=chain_ids[0] if len(chain_ids) == 1 else None,
            chain_ids=chain_ids,
            metadata=metadata,
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


def add_asset_weight_legend(
    figure: Figure,
    entries: Iterable[AssetWeightLegendEntry],
    *,
    chain_icon_resolver: Callable[[int], str | None] = resolve_chain_icon_data_url,
) -> None:
    """Replace a native legend with aligned allocation and identity columns.

    The Plotly legend cannot render images. This layout puts all column headers
    and icon boxes on fixed paper-coordinate centres, so absent curator logos
    reserve an empty but aligned slot.

    :param figure:
        Asset-weight figure whose trace names identify legend entries.
    :param entries:
        Trace metadata, keyed by :attr:`AssetWeightLegendEntry.label`.
    :param chain_icon_resolver:
        Injectable resolver used by tests to avoid external icon downloads.
    """
    entries_by_label = {entry.label: entry for entry in entries}
    row_count = max(len(figure.data), 1)
    row_step = (ROW_TOP_Y - ROW_BOTTOM_Y) / max(row_count - 1, 1)
    icon_height = min(MAX_ICON_HEIGHT, row_step * 0.90)
    figure_height = max(900, (row_count + 2) * 24)

    figure.update_layout(
        showlegend=False,
        width=1800,
        height=figure_height,
        margin={"r": 20},
    )
    figure.update_xaxes(domain=[0, 0.608])

    for label, x, xanchor in (
        ("A", ALLOCATION_X, "center"),
        ("C", CHAIN_X, "center"),
        ("P", PROTOCOL_X, "center"),
        ("C", CURATOR_X, "center"),
        ("Name", NAME_X, "left"),
    ):
        figure.add_annotation(
            x=x,
            y=HEADER_Y,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            xanchor=xanchor,
            yanchor="middle",
            font={"size": HEADER_FONT_SIZE, "color": "#a8b1c1"},
        )

    for index, trace in enumerate(figure.data):
        entry = entries_by_label.get(trace.name)
        y = ROW_TOP_Y - index * row_step
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
            x=ALLOCATION_X,
            y=y,
            xref="paper",
            yref="paper",
            sizex=ALLOCATION_ICON_WIDTH,
            sizey=icon_height,
            xanchor="center",
            yanchor="middle",
            sizing="stretch",
            layer="above",
        )

        chain_ids = _entry_chain_ids(entry) if entry else ()
        chain_icon_size = min(IDENTITY_ICON_WIDTH, 0.036 / len(chain_ids)) if chain_ids else IDENTITY_ICON_WIDTH
        chain_icon_spacing = chain_icon_size * 1.1
        chain_sources = [
            chain_icon_resolver(chain_id)
            for chain_id in chain_ids
        ]
        protocol_source = None
        curator_source = None
        if entry and entry.metadata:
            logos = resolve_vault_legend_logos(entry.metadata)
            protocol_source = local_png_data_url(logos.protocol) if logos.protocol else None
            curator_source = local_png_data_url(logos.curator) if logos.curator else None

        chain_icon_offset = (len(chain_sources) - 1) * chain_icon_spacing / 2
        icon_specs = [
            (CHAIN_X - chain_icon_offset + index * chain_icon_spacing, source, chain_icon_size)
            for index, source in enumerate(chain_sources)
        ]
        icon_specs.extend(
            (
                (PROTOCOL_X, protocol_source, IDENTITY_ICON_WIDTH),
                (CURATOR_X, curator_source, IDENTITY_ICON_WIDTH),
            ),
        )

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
            x=NAME_X,
            y=y,
            xref="paper",
            yref="paper",
            text=trace.name,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font={"size": ROW_FONT_SIZE, "color": "#e5ecf6"},
        )
