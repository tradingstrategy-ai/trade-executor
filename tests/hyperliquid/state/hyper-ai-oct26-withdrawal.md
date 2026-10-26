# Withdrawal checkpoint fixture

`hyper-ai-oct26-withdrawal.json` is a reduced copy of the 26 October 2026
Hyper-AI checkpoint: #1825 has two withdrawal transactions and an ambiguous
second-stage marker, but the final failure/freeze checkpoint was not saved.
The three untouched buys are #1822–#1824.

Historical closed positions, visualisation/statistics, signed transaction
payloads, pricing diagnostics and balance-update details were removed. Open
positions and trade quantities remain so tests exercise real partial-fill and
sibling-repair accounting. Addresses and transaction hashes are public; this
fixture contains no RPC credentials or signing keys. The original downloaded
state and operational logs must not be committed.
