"""Charts export to PNG.

- Render Plotly figures as static PNG images

"""
from io import BytesIO

import plotly.graph_objects as go


def render_plotly_figure_as_image_file(
        figure: go.Figure,
        format: str = "png",
        width: int = 512,
        height: int = 512,
) -> bytes:
    """"Render Plotly figure as a static PNG image.

    See

    - https://plotly.com/python/static-image-export/

    - https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_image.html

    :param format:
        "png" or "svg"
    """
    stream = BytesIO()
    figure.write_image(
        stream,
        format=format,
        engine="kaleido",
        width=width,
        height=height,
    )
    data = stream.getvalue()
    stream.close()
    return data
