"""Charts export to PNG.

- Render Plotly figures as static PNG images

- Used on a frontend for the performance charts, in Discord posts

"""
import logging
import tempfile
import webbrowser
import threading
from io import BytesIO
import asyncio
from pathlib import Path

import plotly.graph_objects as go
from gmpy2.gmpy2 import context
from kaleido import Kaleido


logger = logging.getLogger(__name__)

_kaleido = None
_lock = threading.Lock()


def get_kaleido() -> Kaleido:
    """Create Kaleido rendering backend.

    - Creates a Chrome browser on background

    - `See usage examples <https://github.com/plotly/Kaleido>`__
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

    - Uses Kaleido to render the Plotly figure as a PNG or SVG image, or PDF..
    - Creates Kaleido backend (Chrome browser) on the first call.

    :param format:
        See ``kaleido._fig_tools` module for supported formats.

    :param width:
        Width in pixels

    :param height:
        Height in pixels

    :return:
        Image data encoded as byttes blob.
    """

    assert format in ["png", "svg"], "Format must be png or svg"
    assert isinstance(figure, go.Figure), "Figure must be an instance of plotly.graph_objects.Figure"

    logger.info(
        "render_plotly_figure_as_image_file(): %s %s %s",
        type(figure),
        width,
        height,
    )

    with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
        filename = tmp.name
        kaleido_instance = get_kaleido()

        opts = dict(
            format=format,
            width=width,
            height=height,
        )

        import ipdb ; ipdb.set_trace()

        # kaleido_instance.write_fig(
        #    figure,
        #    filename,
        #    opts=opts,
        # )

        asyncio.run(
            kaleido_instance.write_fig(
                figure,
                filename,
                opts=opts,
            ),
            debug=True,
        )

        logger.info("Kaleido rendering done")
        data = open(filename, "rb").read()

    assert len(data) > 0, "Rendered image data is empty"
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

