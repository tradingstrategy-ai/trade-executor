"""Sort helpers."""
from typing import Iterable, Any


def unique_sort(items: list, key: callable, reverse=False) -> list[Any]:
    """Sort the list, but return the unique results only.

    - The first unique match is returned

    :param items:
        Items to sort

    :param accesor:
        Accessor function

    :param reverse:
        Reverse the result

    """

    accessed = ((key(item), item) for item in items)
    first_pass = sorted(accessed, key=accessed[0], reverse=reverse)
    result = []
    uniq = set()
    for key, item in first_pass:
        if key not in uniq:
            uniq.add(key)
            result.append(item)
    return result




