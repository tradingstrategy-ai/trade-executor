"""Keep fork-test waits within the reviewed IPOR observation age limit.

A fork has no operator publishing new observations. An artificial day-long
wait can therefore expire valid data before the withdrawal is tested.
:func:`~tradeexecutor.strategy.vault_protocol.resolve_vault_protocol_rules`
copies these deployment-specific limits into the shared simulation rules.
"""

#: Ethereum HYPE Basis had a two-hour observation-age limit and no share
#: lock at block 26402309. Its 24-hour artificial wait expired those observations.
#: Limit the artificial holding wait to 30 minutes; the adapter refuses data that
#: are already stale. This does not schedule production observation updates.
#: Keys are ``(chain_id, lower-case vault address)`` and values are seconds.
#: Consumers use
#: :func:`~tradeexecutor.strategy.vault_protocol.get_vault_test_trade_time_skip_seconds`
#: so this contract-specific rule stays out of shared execution code.
SIMULATION_MAX_HOLDING_SECONDS = {
    (1, "0x87088ffff874a4bcc6d13bbc8184e3639947ecde"): 30 * 60,
}
