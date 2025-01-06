from tradeexecutor.state.types import Percent, BPS, IntBPS


def get_slippage_in_bps(
    slippage: Percent,
    max_sane_slippage: Percent = 0.21,
) -> IntBPS:
    """Validate and convert slippage to BPS
    
    :param slippage:
        Slippage in percentage.

        0....1.

    :param max_sane_slippage:
        Maximum sane slippage in percentage, default is 0.05 (5%)

    :return:
        Slippage as 10_000 points.

        0.1% = 100.
    """

    assert 0 < slippage <= max_sane_slippage, f"Slippage tolerance value should be between 0 and {max_sane_slippage}, Got {slippage}"
    return slippage * 10_000
