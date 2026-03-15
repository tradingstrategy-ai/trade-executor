"""Shared formatting helpers for grid search analysis and visualisation."""

from __future__ import annotations

from enum import Enum

import pandas as pd


def normalise_enum_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strip enum class names from DataFrame values for presentation."""
    def enum_to_value(x):
        return x.value if isinstance(x, Enum) else x

    if hasattr(df, "map"):
        return df.map(enum_to_value)
    return df.applymap(enum_to_value)


def build_format_dict(value_cols: list[str], percent_cols: list[str], data_cols: list[str]) -> dict[str, object]:
    """Build the shared DataFrame formatting dictionary for grid-search tables."""
    format_dict = {col: "{:.2f}".format for col in value_cols}
    for col in percent_cols:
        format_dict[col] = "{:.2%}".format
    for col in data_cols:
        format_dict[col] = "{0:g}".format
    return format_dict


def build_gradient_colour_map(values: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map feature values to shades of a single colour."""
    try:
        numeric = [(v, float(v)) for v in values]
        numeric.sort(key=lambda x: x[1])
        sorted_values = [v for v, _ in numeric]
    except (ValueError, TypeError):
        sorted_values = list(values)

    light = (255, 180, 180)
    dark = (139, 0, 0)
    colour_map = {}
    for i, val in enumerate(sorted_values):
        t = i / max(len(sorted_values) - 1, 1)
        r = int(light[0] + t * (dark[0] - light[0]))
        g = int(light[1] + t * (dark[1] - light[1]))
        b = int(light[2] + t * (dark[2] - light[2]))
        colour_map[val] = f"rgb({r},{g},{b})"
    return colour_map, sorted_values


def get_group_colour_palette(num_groups: int) -> list[tuple[int, int, int]]:
    """Return distinct RGB tuples for grouped grid-search charts."""
    base_colours = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (255, 127, 14),
        (148, 103, 189),
        (23, 190, 207),
        (188, 189, 34),
        (227, 119, 194),
    ]
    return [base_colours[i % len(base_colours)] for i in range(num_groups)]


def generate_grey_alpha(num_colors: int) -> list[str]:
    """Generate greyscale colours with increasing alpha for fallback chart lines."""
    colors = []
    for i in range(num_colors):
        red = 33 + 128 * i / num_colors
        color = (red, red, red, red)
        colors.append(f"rgba{color}")
    return colors
