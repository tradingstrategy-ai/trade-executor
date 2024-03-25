"""Sort helpers."""
from typing import Any, Callable

import pandas as pd


def unique_sort(items: list, key: Callable, reverse=False, filter_na=True) -> list[Any]:
    """Sort the list, but return the unique results only.

    - Similar as :py:func:`sorted`

    - The first unique matched value is returned

    :param key:
        Accessor function

    :param items:
        Items to sort

    :param accessor:
        Accessor function

    :param reverse:
        Reverse the result

    :param filter_na:
        If the sorted value is NA, ignore it.

        Use `pandas.isna` to detect.
    """
    accessed = ((key(item), item) for item in items)
    first_pass = sorted(accessed, key=lambda x: x[0], reverse=reverse)
    result = []
    uniq = set()
    for key, item in first_pass:

        if filter_na:
            if pd.isna(key):
                continue

        if key not in uniq:
            uniq.add(key)
            result.append(item)
    return result




