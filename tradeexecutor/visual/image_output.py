"""Charts export to PNG.

- Render Plotly figures as static PNG images

- Used on a frontend for the performance charts, in Discord posts

"""
import logging
import webbrowser
from pathlib import Path

from plotly.graph_objects import Figure


logger = logging.getLogger(__name__)


def render_plotly_figure_as_image_file(
    figure: Figure,
    format: str = "png",
    width: int = 512,
    height: int = 512,
) -> bytes:
    """"Render Plotly figure as a static PNG image.

    - Uses Kaleido to render the Plotly figure as a PNG or SVG image
    - Plotly's Figure.to_image uses Kaleido when it's installed/available

    :param format:
        See ``kaleido._fig_tools` module for supported formats.
        Supporting only png and svg for now.

    :param width:
        Width in pixels

    :param height:
        Height in pixels

    :return:
        Image data encoded as byttes blob.
    """

    assert format in ["png", "svg"], "Format must be png or svg"
    assert isinstance(figure, Figure), "Figure must be an instance of plotly.graph_objects.Figure"

    logger.info(f"render_plotly_figure_as_image_file(): {type(figure)} {width}x{height}")

    data = figure.to_image(format, width, height)
    assert len(data) > 0, "Rendered image data is empty"

    logger.info("Plotly/Kaleido rendering done")

    return data


def open_plotly_figure_in_browser(figure: Figure, height:int = 512, width: int = 512) -> None:
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
