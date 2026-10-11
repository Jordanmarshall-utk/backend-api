# POLARIS instrument safety boundary

POLARIS currently produces **protocol drafts**, not commands that are authorized to run on laboratory hardware.

## States

1. `draft_unvalidated`: text or code has been generated but deterministic checks have not passed.
2. `syntax_validated_draft`: Python syntax and the supplied worklist pass POLARIS preflight checks.
3. `simulated`: the exact immutable draft and worklist have passed the official vendor simulator.
4. `approved`: a named laboratory user has reviewed the simulation, deck map, consumables, volumes, hazards, and checksum.
5. `executing`: a laboratory-side connector has accepted the approved checksum and is reporting telemetry.
6. `completed`, `failed`, or `aborted`: a terminal, audited run state.

Only a separate laboratory-network connector may transition an artifact from `approved` to `executing`. The cloud service must never have a direct network route to an OT-2 or other instrument.

## Current OT-2 support

- The generator emits a portable Python draft and no longer embeds a workstation-specific Jupyter path.
- Deterministic preflight checks cover Python syntax, imports, canonical wells, tube-source capacity, P300 transfer bounds, destination capacity, and duplicate wells.
- A fresh tip is required for each source-to-destination transfer to avoid returning a destination-contacted tip to a stock tube.
- Official Opentrons simulation is optional and is not installed in the standard POLARIS runtime today.
- `execution_supported` is always `false`; generation or successful syntax validation must not set `automation_ready` to `true`.

## Required lab-side connector controls

- Mutual authentication and a device-specific allowlist.
- Human approval bound to the protocol and worklist SHA-256 checksums.
- Idempotency keys and an explicit run state machine.
- Robot identity, deck inventory, calibration, consumables, source volumes, and waste-capacity checks.
- Emergency stop, door/module interlocks, timeouts, and loss-of-connectivity behavior.
- Append-only audit events and telemetry with UTC timestamps.
- No arbitrary Python, filesystem, shell, Jupyter, or generic MCP tool forwarding.

The first Athena MCP gateway is intentionally read-only. Hardware simulation may later be exposed as a bounded tool; physical execution must remain a separately deployed and separately authorized capability.
