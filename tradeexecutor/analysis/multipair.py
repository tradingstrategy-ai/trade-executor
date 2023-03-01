"""Multipair strategy analyses.

Designed for strategies trading > 5 assets.
"""
import pandas as pd


def analyse_multipair(state: State) -> pd.DataFrame:
    """Build an analysis table.

    Create a table where 1 row = 1 trading pair.

    :param state:

    :return:
        Human-readable dataframe.
    """

