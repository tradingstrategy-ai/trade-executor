"""List utilities."""

def get_linearly_sampled_items(lst: list, count: int):
    """Get N items from a list, equally distributed over the index."""
    if len(lst) < count:
        return lst

    step = (len(lst) - 1) / (count - 1)
    indices = [round(i * step) for i in range(count)]
    return [lst[i] for i in indices]