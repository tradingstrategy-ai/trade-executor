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
    Generates a Plotly color palette in hacker CRT green style.

    Parameters:
    num_colors (int): Number of colors to generate.

    Returns:
    list: A list of color strings in hexadecimal format.
    """
    base_green = "#00FF00"  # Base CRT green color
    palette = []

    for i in range(num_colors):
        if i % 2 == 0:
            # Lighter shade (base green)
            palette.append(base_green)
        else:
            # Darker shade (reduce the green channel)
            darker_green = "#{:02X}{:02X}{:02X}".format(0, int(0xFF * 0.6), 0)
            palette.append(darker_green)

    return palette
