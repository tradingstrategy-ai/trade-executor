"""Check for double positions."""
from tradeexecutor.state.state import State


def _get_position_grouping_key(position):
    """Get the duplicate-detection grouping key for a position."""
    pair = position.pair

    # Hypercore vault pair equality intentionally ignores ``pool_address``,
    # but duplicate-position diagnostics must distinguish different vaults.
    # Group these positions by their actual vault address instead.
    if pair.is_hyperliquid_vault():
        return ("hyperliquid_vault", (pair.pool_address or pair.base.address).lower())

    return pair


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


def get_duplicate_position_groups(state: State) -> list[list]:
    """Get position groups that are considered duplicates.

    Hypercore vaults are stricter than generic pairs: any repeated open/frozen
    position for the same vault address is treated as a duplicate, because even
    one stale residual can break later accounting and execution paths.
    """

    grouped_positions = {}
    for position in state.portfolio.get_open_and_frozen_positions():
        grouping_key = _get_position_grouping_key(position)
        grouped_positions.setdefault(grouping_key, []).append(position)

    duplicate_groups = []
    for positions in grouped_positions.values():
        if len(positions) < 2:
            continue

        pair = positions[0].pair
        not_about_to_close = [p for p in positions if not p.is_about_to_close()]
        hypercore_duplicate = pair.is_hyperliquid_vault()
        generic_duplicate = len(not_about_to_close) >= 2
        if generic_duplicate or hypercore_duplicate:
            duplicate_groups.append(positions)

    return duplicate_groups


def _print_duplicate_position_group(positions: list, printer=print) -> None:
    """Print diagnostics for one duplicate-position group."""
    pair = positions[0].pair

    if pair.is_hyperliquid_vault():
        printer(
            "Warning: Hypercore vault pair "
            f"{pair} has multiple open/frozen positions: {len(positions)}. "
            "This usually means a residual dust position stayed open and a later cycle "
            "opened a second live position for the same vault."
        )
    else:
        printer(f"Warning: pair {pair} has multiple open positions: {len(positions)}")

    for position in positions:
        printer(f"Position {position}: {_format_position_diagnostics(position)}")


def check_double_position(state: State, printer=print, crash=False) -> bool:
    """Check that we do not have multiple positions open for the same trading pair.

    :param printer:
        Replace with logger.error() for live execution

    :param crash:
        Safety crash - do not allow continue beyond this point if we detect double positions.

    :return:
        True if there are double positions
    """
    double_positions = False
    duplicate_groups = get_duplicate_position_groups(state)

    for positions in duplicate_groups:
        pair = positions[0].pair
        double_positions = True
        _print_duplicate_position_group(positions, printer=printer)

        if crash:
            if pair.is_hyperliquid_vault():
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
