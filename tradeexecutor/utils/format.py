from typing import Optional


def calculate_percentage(
    number1: float, 
    number2: float, 
    default_value=None,
) -> Optional[float]:
    """Get percentage of number1 to number2, return default_value if division by zero
    """
    return number1 / number2 if number2 else default_value