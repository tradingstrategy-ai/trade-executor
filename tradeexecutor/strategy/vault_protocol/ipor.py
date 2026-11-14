"""Simulation waits for reviewed IPOR external-state deployments."""

#: The Ethereum HYPE Basis route has a two-hour observation limit and no share
#: lock at Ethereum block 26402309. Its 24-hour artificial wait expired the feed.
#: Limit the artificial holding wait to 30 minutes; the adapter refuses data that
#: are already stale. This does not schedule production external-state updates.
SIMULATION_MAX_HOLDING_SECONDS = {
    (1, "0x87088ffff874a4bcc6d13bbc8184e3639947ecde"): 30 * 60,
}
