"""Sort helpers."""
from typing import Any, Callable

import pandas as pd


def unique_sort(items: list, key: Callable, reverse=False, filter_na=True, check_number=True) -> list[Any]:
    """Sort the list, but return the unique results only.

    - Used for sorting the best grid search results (by CAGR, Sharpe)

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

    :param check_number:
        Make sure all values in items list are numbers.

        A helper check to avoid malformed data leaking through.
    """

    accessed = [(key(item), item) for item in items]

    if check_number:
        for idx, tpl in enumerate(accessed, start=1):
            val, orig_obj = tpl
            assert type(val) in (float, int,), f"Received non-number in unique_sort(): {type(val)}: {val} (list item #{idx})"

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




