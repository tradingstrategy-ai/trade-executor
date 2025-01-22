"""Various palettes for charts."""


def generate_sepia_palette(n=40):
    colors = []
    # Base sepia colors from light to dark
    sepia_base = [
        '#fff4e6',  # Very light sepia
        '#ffe4c4',  # Bisque
        '#deb887',  # Burlywood
        '#d2b48c',  # Tan
        '#bc8f8f',  # Rosy brown
        '#a0522d',  # Sienna
        '#8b4513',  # Saddle brown
        '#654321',  # Dark brown
    ]

    # Create variations by interpolating between base colors
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    import numpy as np

    # Create a colormap from our sepia colors
    sepia_cmap = LinearSegmentedColormap.from_list('sepia', sepia_base)

    # Generate n colors
    values = np.linspace(0, 1, n)
    colors = [to_hex(sepia_cmap(v)) for v in values]

    return colors


def hacker_crt_green_palette(num_colors):
    """
    Generates a Plotly color palette in hacker CRT green style with varying brightness.

    Parameters:
    num_colors (int): Number of colors to generate.

    Returns:
    list: A list of color strings in hexadecimal format.
    """
    palette = []

    for i in range(num_colors):
        if i % 2 == 0:
            # Brighter green with varying intensity
            intensity = 255 - int((i / num_colors) * 100)  # Adjust intensity based on position
            brighter_green = f"#00{intensity:02X}00"
            palette.append(brighter_green)
        else:
            # Darker green (fixed for contrast)
            darker_green = "#007700"
            palette.append(darker_green)

    return palette

