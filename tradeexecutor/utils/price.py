"""Price insight helpers."""


def is_legit_price_value(usd_unit_price: float) -> bool:
    """Avoid taking positions in trading pairs where pricing in wacky.

    Pricing unit might be too small or too large, causing

    - Displaying issues

    - Floating point rounding issues

    Usually legit trading pairs do not have such price units.
    """
    return (usd_unit_price < 0.000000001) or (usd_unit_price > 100_000)