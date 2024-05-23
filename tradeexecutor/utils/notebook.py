"""Jupyter notebook utilities for backtesting."""
import enum
import logging

import pandas as pd
import matplotlib_inline


class OutputMode(enum.Enum):
    """What is the output mode for the notebook visualisations.

    Interactive visualisations work only on the HTML pages
    that are able to load Plotly.js JavaScripts.

    For examples see :py:func:`setup_charting_and_output`.
    """

    #: Output charts as static images
    static = "static"

    #: Output charts as interactive Plotly.js visualisations
    interactive = "interactive"


def setup_charting_and_output(
    mode: OutputMode=OutputMode.interactive,
    image_format="svg",
    max_rows=1000,
    width=1500,
    height=1500,
):
    """Sets charting and other output options for Jupyter Notebooks.

    Interactive charts are better for local development, but are not compatible with most web-based notebook viewers.

    - `Set Quantstats chart to SVG output and for high-resolution screens <https://stackoverflow.com/questions/74721731/how-to-generate-svg-images-using-python-quantstat-library>`__

    - Mute common warnings like `Matplotlib font loading <https://stackoverflow.com/questions/42097053/matplotlib-cannot-find-basic-fonts/76136516#76136516>`__

    - `Plotly discussion <https://github.com/plotly/plotly.py/issues/931>`__

    Example how to set up default interactive output settings. Add early of your notebook do:

    .. code-block:: python

        # Set Jupyter Notebook output mode parameters.
        # For example, table max output rows is lifted from 20 to unlimited.
        from tradeexecutor.utils.notebook import setup_charting_and_output
        setup_charting_and_output()

    Example how to set up static image rendering:

        # Set charts to static image output, 1500 x 1000 pixels
        from tradeexecutor.utils.notebook import setup_charting_and_output, OutputMode
        setup_charting_and_output(OutputMode.static, image_format="png", width=1500, height=1000)

    :param mode:
        What kind of viewing context we have for this notebook output

    :param image_format:
        Do we do SVG or PNG.

        SVG is better, but Github inline viewer cannot display it in the notebooks.

    :param max_rows:
        Do we remove the ``max_rows`` limitation from Pandas tables.

        Default 20 is too low to display summary tables.
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
        
        current_renderer = pio.renderers[image_format]
        # Have SVGs default pixel with
        current_renderer.width = width
        current_renderer.height = height

    # TODO: Currently we do not reset interactive mode if the notebook has been run once
    # If you run setup_charting_and_output(offline) once you are stuck offline

    if max_rows:
        pd.set_option('display.max_rows', max_rows)
