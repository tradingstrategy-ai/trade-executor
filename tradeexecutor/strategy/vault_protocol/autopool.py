"""Autopool timing for disposable-fork vault test trades."""

# Chainlink feeds can expire after 1h on a static fork; use a 30-minute simulation cycle.
SIMULATION_CYCLE_SECONDS = 30 * 60
