"""Keep Autopool simulation waits within the oracle price validity period."""

#: Chainlink prices can expire after one hour. Anvil's local chain copy receives
#: no new prices, so limit the test's deposit-to-withdrawal wait to 30 minutes.
#: The contract still checks price age; this does not set the live strategy cycle.
SIMULATION_CYCLE_SECONDS = 30 * 60
