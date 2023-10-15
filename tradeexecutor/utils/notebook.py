"""Jupyter notebook utilities for backtesting."""
import enum
import logging


import matplotlib_inline


class OutputMode(enum.Enum):
    """What is the output mode for the notebook visualisations.

    Interactive visualisations work only on the HTML pages
    that are able to load Plotly.js JavaScripts.
    """

    #: Output charts as static images
    static = "static"

    #: Output charts as interactive Plotly.js visualisations
    interactive = "interactive"


def setup_charting_and_output(mode: OutputMode=OutputMode.interactive, image_format="svg"):
    """Sets up Jupyter Notebook based charting.

    - `Set Quantstats chart to SVG output and for high-resolution screens <https://stackoverflow.com/questions/74721731/how-to-generate-svg-images-using-python-quantstat-library>`__

    - Mute common warnings like `Matplotlib font loading <https://stackoverflow.com/questions/42097053/matplotlib-cannot-find-basic-fonts/76136516#76136516>`__

    Example:

    .. code-block:: python

        # Set Jupyter Notebook output mode parameters
        from tradeexecutor.backtest.notebook import setup_charting_and_output
        setup_charting_and_output()

    :param mode:
        What kind of viewing context we have for this notebook output

    """

    # Get rid of findfont: Font family 'Arial' not found.
    # when running a remote notebook on Jupyter Server on Ubuntu Linux server
    # https://stackoverflow.com/questions/42097053/matplotlib-cannot-find-basic-fonts/76136516#76136516
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    # Render charts from quantstats in high resolution
    # https://stackoverflow.com/questions/74721731/how-to-generate-svg-images-using-python-quantstat-library
    matplotlib_inline.backend_inline.set_matplotlib_formats(image_format)

    # Set Plotly to offline (static image mode)
    if mode == OutputMode.static:

         # https://stackoverflow.com/a/52956402/315168
        from plotly.offline import init_notebook_mode
        init_notebook_mode()

        # https://stackoverflow.com/a/74609837/315168
        import plotly.io as pio
        pio.kaleido.scope.default_format = image_format

        # https://plotly.com/python/renderers/#overriding-the-default-renderer
        pio.renderers.default = image_format
        svg_renderer = pio.renderers[image_format]
        # Have SVGs default pixel with
        svg_renderer.width = 1500
