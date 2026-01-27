<div align="center">

# 🚀 RFSN Controller

**Autonomous Code Repair Agent with Hierarchical Planning & Serial Decision Architecture**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-102%20passing-brightgreen.svg)](#testing)
[![CGW Architecture](https://img.shields.io/badge/CGW-Serial%20Decisions-purple.svg)](#cgw-mode)
[![Phase 4](https://img.shields.io/badge/Phase%204-Complete-green.svg)](#hierarchical-planner)

*Fix bugs autonomously. One decision at a time. Safety guaranteed.*

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 CGW Serial Decision Mode

One decision per cycle. No parallel chaos.

- **Thalamic Gate** arbitration
- **Forced signal** override for safety
- **Event replay** for debugging
- **Seriality verification**

</td>
<td width="50%">

### 🛡️ Plan Gate (Hard Safety)

Planner proposes, Gate authorizes.

- **Step type allowlist** enforcement
- **Shell injection detection**
- **Path validation** (workspace-only)
- **Budget enforcement** (max steps)

</td>
</tr>
<tr>
<td width="50%">

### 📋 Hierarchical Planner v4

High-level goal decomposition with learning.

- **Failure fingerprinting** (categorization)
- **Thompson Sampling** strategy selection
- **Quarantine lane** (anti-regression)
- **Proposal-space learning** only

</td>
<td width="50%">

### ⚡ Multi-Model Ensemble

Active-active LLM failover.

- DeepSeek V3 primary
- Gemini 2.0 Flash fallback
- Thompson Sampling model selection
- Consensus voting on patches

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RFSN Controller                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐    │
│   │   Planner   │───▶│  Plan Gate  │───▶│   Controller Loop   │    │
│   │ (proposes)  │    │ (validates) │    │     (executes)      │    │
│   └─────────────┘    └─────────────┘    └─────────────────────┘    │
│         ▲                                         │                  │
│         │                                         ▼                  │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                     Learning Layer                           │  │
│   │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │  │
│   │  │ Fingerprint│  │   Bandit   │  │    Quarantine      │    │  │
│   │  │ (classify) │  │  (select)  │  │ (anti-regression)  │    │  │
│   │  └────────────┘  └────────────┘  └────────────────────┘    │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Safety Guarantees

| Guarantee | Implementation |
|-----------|----------------|
| **Planner never executes** | Produces JSON data only |
| **Gate has veto power** | Cannot be bypassed |
| **Learning cannot weaken gates** | Proposal space only |
| **Serial execution** | One mutation at a time |
| **No regressions** | Quarantine auto-blocks |

---

## 🚀 Quick Start

```bash
# Install
pip install -e .

# Set API keys
export DEEPSEEK_API_KEY="sk-..."
export GEMINI_API_KEY="..."

# Run (CGW serial decision mode)
python -m rfsn_controller.cli --repo https://github.com/user/repo --cgw-mode

# Run with hierarchical planner
python -m rfsn_controller.cli --repo https://github.com/user/repo --planner-mode v4
```

---

## 🛡️ Hierarchical Planner

The Phase 4 hierarchical planner adds safe learning to the controller:

```python
from rfsn_controller.gates import PlanGate, PlanGateConfig
from rfsn_controller.learning import LearnedStrategySelector
from rfsn_controller.controller_loop import ControllerLoop

# Setup with safety config
config = PlanGateConfig(max_steps=10)
gate = PlanGate(config)
selector = LearnedStrategySelector()
loop = ControllerLoop(gate=gate, learning=selector)

# Get AI recommendation based on failure patterns
rec = selector.recommend(failing_tests=["test_auth_flow"])
print(f"Strategy: {rec.strategy}, Confidence: {rec.confidence:.0%}")

# Run plan (gate validates every step)
plan = {
    "plan_id": "fix_auth",
    "steps": [
        {"id": "s1", "type": "read_file", "inputs": {"file": "auth.py"}, "expected_outcome": "understand"},
        {"id": "s2", "type": "apply_patch", "inputs": {...}, "expected_outcome": "fix"},
        {"id": "s3", "type": "run_tests", "inputs": {}, "expected_outcome": "pass"},
    ]
}

result = loop.run_plan(plan)
print(f"Success: {result.success}, Steps: {result.steps_succeeded}/{result.steps_executed}")

# Update learning (anti-regression)
selector.update(rec, success=result.success)
```

### Allowed Step Types

```python
DEFAULT_ALLOWED_STEP_TYPES = {
    # Read-only
    "search_repo", "read_file", "analyze_file", "list_directory", "grep_search",
    # Code modification (sandboxed)
    "apply_patch", "add_test", "refactor_small", "fix_import", "fix_typing",
    # Verification
    "run_tests", "run_lint", "check_syntax", "validate_types",
    # Coordination
    "wait", "checkpoint", "replan",
}
```

---

## 🧠 CGW Mode

The **Conscious Global Workspace (CGW)** architecture enforces serial decision-making:

```
Decide → Commit → Execute → Report → Next Cycle
```

```python
from cgw_ssl_guard.coding_agent import CodingAgentRuntime, AgentConfig

runtime = CodingAgentRuntime(config=AgentConfig(goal="Fix tests"))
result = runtime.run_until_done()

print(result.summary())
# [SUCCESS] FINALIZE after 5 cycles. Tests passing: True.
```

---

## 📁 Project Structure

```
├── cgw_ssl_guard/           # CGW/SSL Guard Core
│   ├── coding_agent/        # Serial Decision Coding Agent
│   │   ├── action_types.py
│   │   ├── proposal_generators.py
│   │   ├── executor.py
│   │   ├── coding_agent_runtime.py
│   │   ├── config.py            # YAML/JSON configuration
│   │   ├── cli.py               # CLI entry point
│   │   └── integrated_runtime.py
│   ├── thalamic_gate.py
│   ├── event_bus.py
│   └── monitors.py
│
├── rfsn_controller/         # Main Controller
│   ├── controller.py        # 2600+ line repair loop
│   ├── controller_loop.py   # NEW: Serial execution with planner
│   ├── gates/               # NEW: Safety gates
│   │   ├── __init__.py
│   │   └── plan_gate.py     # Hard safety enforcement
│   ├── learning/            # NEW: Proposal-space learning
│   │   ├── fingerprint.py   # Failure categorization
│   │   ├── strategy_bandit.py
│   │   ├── quarantine.py
│   │   └── learned_strategy_selector.py
│   ├── planner_v2/          # Planner system
│   ├── qa/                   # QA/verification
│   ├── buildpacks/          # Language support
│   └── cgw_bridge.py        # CGW integration
│
├── tests/                   # Test Suite (102 tests)
│   ├── cgw/                 # CGW tests (18)
│   ├── test_phase2.py       # Phase 2 tests (37)
│   ├── rfsn_controller/
│   │   └── test_phase4.py   # Phase 4 tests (25)
│   └── ...
│
└── docs/                    # Documentation
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run Phase 4 tests (hierarchical planner)
pytest tests/rfsn_controller/test_phase4.py -v

# Run CGW tests
pytest tests/cgw/ -v

# Quick validation
python -c "
from rfsn_controller.gates import PlanGate
from rfsn_controller.learning import LearnedStrategySelector
from rfsn_controller.controller_loop import ControllerLoop
print('✓ Phase 4 imports successful')
"
```

---

## 🌐 Language Support

| Language | Buildpack | Tools |
|----------|-----------|-------|
| **Python** | `python_pack` | pip, uv, pytest, nose |
| **Node.js** | `node_pack` | npm, yarn, pnpm, jest |
| **Go** | `go_pack` | go mod, go test |
| **Rust** | `rust_pack` | cargo |
| **C/C++** | `cpp_pack` | gcc, cmake, make |
| **Java** | `java_pack` | maven, gradle |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [CGW_CODING_AGENT.md](docs/CGW_CODING_AGENT.md) | CGW architecture guide |
| [USAGE_GUIDE.md](docs/USAGE_GUIDE.md) | Full usage guide |
| [FEATURE_MODE.md](docs/FEATURE_MODE.md) | Feature engineering mode |
| [DOCKER_SANDBOX.md](docs/DOCKER_SANDBOX.md) | Docker sandbox setup |

---

## ⚙️ Configuration

```bash
# Model selection
--model deepseek-chat

# CGW mode
--cgw-mode
--max-cgw-cycles 50

# Planner (v4 includes learning)
--planner-mode v4
--max-plan-steps 12

# Learning
--learning-db ./learning.db
--policy-mode bandit
--quarantine-threshold 0.3

# Parallel patches
--parallel-patches
--ensemble-mode
```

---

## 🔒 Security

- All code runs in isolated Docker containers
- No host execution by default
- **PlanGate** validates every step before execution
- **Shell injection detection** blocks dangerous commands
- APT package whitelisting
- Command allowlisting
- Patch size limits

See [SECURITY.md](SECURITY.md) for details.

---

## 📄 License

MIT License. See [LICENSE](LICENSE).

---

<div align="center">

**Built for autonomous code repair at scale.**

*Planner proposes. Gate authorizes. Controller executes.*

</div>
