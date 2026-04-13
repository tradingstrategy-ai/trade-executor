"""Check for double positions."""
from tradeexecutor.state.state import State


def _format_position_diagnostics(position) -> str:
    """Format one position for duplicate-position diagnostics."""
    pair = position.pair
    status = "frozen" if position.is_frozen() else "open"
    details = [
        f"position_id={position.position_id}",
        f"status={status}",
        f"quantity={position.get_quantity(planned=False)}",
        f"planned_quantity={position.get_quantity(planned=True)}",
        f"has_planned_trades={position.has_planned_trades()}",
        f"is_about_to_close={position.is_about_to_close()}",
        f"can_be_closed={position.can_be_closed()}",
    ]
    if pair.is_hyperliquid_vault():
        details.append(f"vault_name={pair.get_vault_name() or pair.get_ticker()}")
        details.append(f"vault_address={pair.pool_address or pair.base.address}")
    return ", ".join(details)


def check_double_position(state: State, printer=print, crash=False) -> bool:
    """Check that we do not have multiple positions open for the same trading pair.

    :param printer:
        Replace with logger.error() for live execution

    :param crash:
        Safety crash - do not allow continue beyond this point if we detect double positions.

    :return:
        True if there are double positions
    """
    # Warn about pairs appearing twice in the portfolio
    double_positions = False
    pairs = {p.pair for p in state.portfolio.get_open_and_frozen_positions()}
    for pair in pairs:

        positions = [p for p in state.portfolio.get_open_and_frozen_positions() if p.pair == pair]
        if len(positions) < 2:
            continue

        not_about_to_close = [p for p in positions if not p.is_about_to_close()]
        hypercore_duplicate = pair.is_hyperliquid_vault()
        generic_duplicate = len(not_about_to_close) >= 2

        if generic_duplicate or hypercore_duplicate:
            if hypercore_duplicate:
                printer(
                    "Warning: Hypercore vault pair "
                    f"{pair} has multiple open/frozen positions: {len(positions)}. "
                    "This usually means a residual dust position stayed open and a later cycle "
                    "opened a second live position for the same vault."
                )
            else:
                printer(f"Warning: pair {pair} has multiple open positions: {len(positions)}")

            for p in positions:
                printer(f"Position {p}: {_format_position_diagnostics(p)}")

            double_positions = True

            if crash:
                if hypercore_duplicate:
                    raise AssertionError(
                        f"Duplicate Hypercore vault positions detected for pair {pair}. "
                        "This usually means a stale Hypercore dust residual was left open and a later cycle "
                        "opened a second live position for the same vault. "
                        "Diagnose the duplicate positions from the logs and repair or close the residual state "
                        "before continuing."
                    )
                raise AssertionError(
                    f"Double positions detected for pair {pair} - crashing for safety reasons.\n"
                    f"Positions: {positions}\n"
                    "Diagnose what is causing the double position creation and manually clean up with close-position CLI command.\n"
                    "See logs for more details."
                )

    return double_positions
