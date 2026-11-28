# Faction Analysis Sidecar

Read `AGENTS.md` for the purpose, the decisions and the invariants of this sidecar, and `README.md` for
the environment variables, the run commands, the tables and the formulas.

Telemetry must stay low-cardinality: fixed stage names and counts only. It never carries a high-cardinality
value such as a `did`, a `handle`, a `faction_id`, a `uri`, query text or an exception message.
