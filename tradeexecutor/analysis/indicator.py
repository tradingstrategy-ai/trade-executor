"""Indicator diagnostics."""
import pandas as pd

from tradeexecutor.strategy.pandas_trader.indicator_decorator import IndicatorRegistry


def display_indicators(
    indicators: IndicatorRegistry,
    html=True,
) -> pd.DataFrame:
    """Create human-readable summary of indicators.

    Example:

    .. code-block:: python

        from tradeexecutor.analysis.indicator import display_indicators

        indicators = IndicatorRegistry()
        # ...
        df = display_indicators(indicators)
        display(df)

        indicator_data = calculate_and_load_indicators_inline(
            strategy_universe=strategy_universe,
            create_indicators=indicators.create_indicators,
            parameters=parameters,
        )

    :param html:
        Prepare HTML output with newlines

    :return:
        Human-readable table
    """

    assert isinstance(indicators, IndicatorRegistry)

    # Display indicators diagnostics table
    indicators_debug_df = indicators.get_diagnostics()
    print(f"We have {len(indicators_debug_df)} indicators:")
    def replace_comma_with_newline(text):
        # print(text)
        if isinstance(text, str):
            return text.replace(',', '<br>')
        return text
    if html:
        indicators_debug_df = indicators_debug_df.map(replace_comma_with_newline)
    return indicators_debug_df

