# CGW Coding Agent Architecture

> Serial decision controller for autonomous software engineering

## Overview

The CGW (Conscious Global Workspace) Coding Agent implements a **serial decision architecture** that enforces:

- **One decision per cycle** (single-slot workspace)
- **No parallel decisions** (thalamic gate arbitration)
- **Atomic commit only** (CGW runtime)
- **Blocking execution** (no tool overlap)

This is a control system, not a chatbot.

---

## Quick Start

```python
from cgw_ssl_guard.coding_agent import CodingAgentRuntime, AgentConfig

# Configure the agent
config = AgentConfig(
    goal="Fix failing tests",
    max_cycles=50,
    max_patches=10,
)

# Create and run
runtime = CodingAgentRuntime(config=config)
result = runtime.run_until_done()

print(result.summary())
# [SUCCESS] FINALIZE after 5 cycles (234.5ms). Tests passing: True.
```

---

## Architecture

```
Decision Layer (CGW + Gate)
│
├── Proposal Generators → Candidates
│   ├── SafetyProposalGenerator (ABORT on trigger)
│   ├── PlannerProposalGenerator (next step from goal)
│   └── IdleProposalGenerator (fallback)
│
├── Thalamic Gate → Single Winner
│   └── Forced queue checked first
│
├── CGW Runtime → Atomic Commit
│   └── One slot, atomic swap
│
└── Blocking Executor → Action Execution
    └── Returns results for next cycle
```

---

## One Full Cycle

```
1. COLLECT PROPOSALS
   Each generator analyzes context and submits Candidates

2. GATE SELECTION
   Gate scores candidates (saliency + urgency + surprise)
   Forced signals bypass competition

3. CGW COMMIT
   Winner committed atomically to workspace
   Event emitted: CGW_COMMIT

4. EXECUTE (BLOCKING)
   Executor runs action synchronously
   No other action can run until complete

5. UPDATE CONTEXT
   Results update proposal context

6. → NEXT CYCLE
```

---

## Action Types

| Action | Category | Description |
|--------|----------|-------------|
| `RUN_TESTS` | Execution | Run test suite |
| `ANALYZE_FAILURE` | Analysis | Parse test failures |
| `GENERATE_PATCH` | Analysis | Request LLM patch |
| `APPLY_PATCH` | Modification | Apply diff to codebase |
| `VALIDATE` | Validation | Run validation checks |
| `FINALIZE` | Terminal | End successfully |
| `ABORT` | Terminal | End with failure |

---

## Forced Signal Override

Safety-critical signals bypass competition:

```python
# Inject abort that wins regardless of other candidates
runtime.inject_forced_signal(CodingAction.ABORT, "max_patches_exceeded")
```

This is the "non-gameable override" for safety conditions.

---

## Seriality Verification

```python
# After running, verify the invariant was maintained
assert runtime.verify_seriality()  # True if no cycle had >1 commit
```

The `SerialityMonitor` tracks commits per cycle and can detect violations.

---

## Integration with RFSN Controller

Use the CGW bridge to integrate with existing controller:

```python
from rfsn_controller.cgw_bridge import CGWControllerBridge, BridgeConfig

config = BridgeConfig(
    github_url="https://github.com/user/repo",
    test_cmd="pytest -q",
)

bridge = CGWControllerBridge(config, sandbox=sandbox)
result = bridge.run()

print(result["cycles_executed"])
print(result["seriality_maintained"])
```

---

## Event Log for Replay

All decisions emit events for deterministic replay:

```python
event_log = bridge.get_event_log()
# [
#   {"event": "GATE_SELECTION", "payload": {...}},
#   {"event": "CGW_COMMIT", "payload": {...}},
#   ...
# ]
```

---

## Testing

```bash
pytest tests/test_cgw_coding_agent.py -v
# 22 tests covering seriality, forced signals, execution, events, workflow
```
