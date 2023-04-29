"""Jupyter notebook utilities for backtesting."""
import logging


import matplotlib_inline


def setup_charting_and_output():
    """Sets up Jupyter Notebook based charting.

    - `Set Quantstats chart to SVG output and for high-resolution screens <https://stackoverflow.com/questions/74721731/how-to-generate-svg-images-using-python-quantstat-library>`__

    - Mute common warnings like `Matplotlib font loading <https://stackoverflow.com/questions/42097053/matplotlib-cannot-find-basic-fonts/76136516#76136516>`__

    Example:

    .. code-block:: python

        # Set Jupyter Notebook output mode parameters
        from tradeexecutor.backtest.notebook import setup_charting_and_output
        setup_charting_and_output()
    """

    # Get rid of findfont: Font family 'Arial' not found.
    # when running a remote notebook on Jupyter Server on Ubuntu Linux server
    # https://stackoverflow.com/questions/42097053/matplotlib-cannot-find-basic-fonts/76136516#76136516
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    # Render charts from quantstats in high resolution
    # https://stackoverflow.com/questions/74721731/how-to-generate-svg-images-using-python-quantstat-library
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


