"""Monkey-patch Plotly 6.x bug: FigureWidget showing nanoseconds instead of dates.

- Workaround bug https://github.com/plotly/plotly.py/issues/5210
"""
import warnings
from importlib.metadata import version, PackageNotFoundError
import datetime

from packaging.version import Version


try:
    pkg_version = version("plotly")
except PackageNotFoundError:
    pkg_version = None


if (pkg_version is not None) and Version(pkg_version) <= Version("7.0.0"):

    import numpy
    import pandas
    from plotly.graph_objs import Figure

    def fix_trace_x_axis_dates(self: Figure):
        for trace in self.data:
            if not hasattr(trace, "x"):
                continue
            item = trace.x[0]
            # Detect datetime64 and convert it to native Python datetime that show() can handle
            if isinstance(trace.x, numpy.ndarray):
                if isinstance(item, numpy.datetime64):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        trace.x = pandas.Series(trace.x).dt.to_pydatetime().tolist()


    # Run in the monkey patch,
    # so that traces are fixed when fig.show() is called
    _old_show = Figure.show
    def _new_show(self: Figure, *args, **kwargs):
        fix_trace_x_axis_dates(self)
        _old_show(self, *args, **kwargs)
    Figure.show = _new_show