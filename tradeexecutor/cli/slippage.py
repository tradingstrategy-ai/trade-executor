"""Slippage tolerance configuration for strategies."""

import logging

from eth_defi.confirmation import logger
from tradeexecutor.strategy.strategy_module import StrategyModuleInformation


logger = logging.getLogger(__name__)

def configure_max_slippage_tolerance(
    max_slippage: float,
    mod: StrategyModuleInformation,
    default_max_slippage=0.0050,
):
    """Figure out what slippage tolerance we should use for the execution.

    - Command line value

    - Strategy module `Parameters` class

    :param max_slippage:
        From command line

    :param mod:
        From strategy module

    """

    if not max_slippage:
        # Read max slippage from the strategy parameters if available
        parameters = mod.parameters
        if parameters:
            if parameters.slippage_tolerance:
                assert type(parameters.slippage_tolerance) == float
                assert 0.0005 <= parameters.slippage_tolerance < 0.025, f"Slippage tolerance is {parameters.slippage_tolerance * 100} % - check if the value is sane"
                max_slippage = parameters.slippage_tolerance

    if not max_slippage:
        logger.info("Slippage tolerance not configured, using default %f", default_max_slippage)
        max_slippage = default_max_slippage

    logger.info("Using slippage tolerance %f %%", max_slippage * 100)

    return max_slippage
