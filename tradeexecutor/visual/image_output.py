"""Charts export to PNG.

- Render Plotly figures as static PNG images

- Used on a frontend for the performance charts, in Discord posts

"""
from io import BytesIO

import plotly.graph_objects as go
import webbrowser
from pathlib import Path


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

    assert format in ["png", "svg"], "Format must be png or svg"

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


def open_plotly_figure_in_browser(figure: go.Figure, height:int = 512, width: int = 512) -> None:
    """Open Plotly figure in a browser. Useful for debugging.

    See https://stackoverflow.com/a/74619515/315168
    
    :param figure:
        Plotly figure
    """
    png_data = render_plotly_figure_as_image_file(figure, height=height, width=width)
    path = Path("/tmp/test-image.png")
    with open(path, "wb") as out:
        out.write(png_data)

    webbrowser.open(f"file://{path.as_posix()}")


def open_bytes_in_browser(data: bytes, format="png") -> None:
    """Open bytes in a browser. Useful for debugging.

    :param data:
        bytes data to be used to create an image
    """

    path = Path(f"/tmp/test-image.{format}")
    with open(path, "wb") as out:
        out.write(data)

    webbrowser.open(f"file://{path.as_posix()}")

