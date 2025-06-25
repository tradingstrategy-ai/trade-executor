"""Charts export to PNG.

- Render Plotly figures as static PNG images

- Used on a frontend for the performance charts, in Discord posts

"""
import threading
import warnings
from io import BytesIO

import plotly.graph_objects as go
import webbrowser
from pathlib import Path

from kaleido import Kaleido

_kaleido = None
_lock = threading.Lock()


def get_kaleido() -> Kaleido:
    """Create Kaleido rendering backend.

    - Creates a Chrome browser on background

    - `See usage <https://github.com/plotly/Kaleido/blob/master/src/py/README.md#usage-examples>`__
    """
    global _kaleido
    with _lock:
        if _kaleido is None:
            _kaleido = Kaleido()

    return _kaleido


def render_plotly_figure_as_image_file(
    figure: go.Figure,
    format: str = "png",
    width: int = 512,
    height: int = 512,
) -> bytes:
    """"Render Plotly figure as a static PNG image.

    - Uses Kaleido to render the Plotly figure as a PNG or SVG image.
    - Creates Kaleido backend (Chrome browser) on the first call.

    :param format:
        "png" or "svg"
    """

    assert format in ["png", "svg"], "Format must be png or svg"

    stream = BytesIO()

    _kaleido = get_kaleido()
    opts = dict(
        format=format,
        width=width,
        height=height,
    )
    _kaleido.write_fig(
        figure,
        stream,
        opts=opts,
    )

    data = stream.getvalue()
    assert len(data) > 0, "Rendered image data is empty"
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

