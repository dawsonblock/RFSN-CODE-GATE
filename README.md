<div align="center">

# 🚀 RFSN Controller

**Autonomous Code Repair Agent with Hierarchical Planning & Serial Decision Architecture**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
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

## 🚀 Quick Start

```bash
# Install
pip install -e .

# Set API keys
export DEEPSEEK_API_KEY="sk-..."
export GEMINI_API_KEY="..."

# Run with Docker (recommended)
python -m rfsn_controller.cli --repo https://github.com/user/repo --test "pytest"

# Run without Docker (local execution)
python -m rfsn_controller.cli --repo ./my-repo --test "pytest" --unsafe-host-exec

# Run with CGW serial decision mode
python -m rfsn_controller.cli --repo https://github.com/user/repo --cgw-mode

# Run with hierarchical planner
python -m rfsn_controller.cli --repo https://github.com/user/repo --planner-mode v4
```

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

## ⚡ Speed Optimizations

<table>
<tr>
<td width="50%">

### 🐳 Docker Warm Pool

Pre-warmed containers eliminate cold-start delays.

- **2-5 second savings** per test run
- Automatic container reuse
- Background pre-warming

</td>
<td width="50%">

### 🧠 Semantic Cache

Embedding-based cache for smarter LLM response reuse.

- **20-40% higher** cache hit rates
- Similarity threshold tuning
- TF-IDF fallback mode

</td>
</tr>
<tr>
<td width="50%">

### 🔀 Parallel LLM Calls

Concurrent patch generation across models.

- **2-3x faster** patch phase
- Multi-temperature sampling
- Best-response selection

</td>
<td width="50%">

### 🎯 Incremental Testing

Run only tests affected by changes.

- **50-90% faster** test runs
- Import graph analysis
- Focused pytest commands

</td>
</tr>
</table>

| Module | Impact | Description |
|--------|--------|-------------|
| `docker_pool` | 2-5s/run | Warm container reuse |
| `semantic_cache` | +40% hits | Embedding similarity cache |
| `prompt_compression` | -30% tokens | Reduce prompt size |
| `streaming_validator` | Early exit | Abort invalid responses |
| `action_store` | Learning | Track outcomes, suggest recovery |
| `speculative_exec` | Preload | Predict and pre-compute next steps |
| `incremental_testing` | 50-90% faster | Affected test selection |

---

## 🐳 Docker vs Local Execution

### Docker Mode (Default)

```bash
# Full isolation, recommended for untrusted repositories
python -m rfsn_controller.cli --repo https://github.com/user/repo --test "pytest"
```

### Local Mode (`--unsafe-host-exec`)

```bash
# Direct execution on host, no Docker required
# Only use with trusted local repositories
python -m rfsn_controller.cli --repo ./my-trusted-repo --test "pytest" --unsafe-host-exec
```

**When to use `--unsafe-host-exec`:**

- Docker daemon not running
- Local development/testing
- Trusted repositories only
- Faster patch verification

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
│   ├── thalamic_gate.py
│   ├── event_bus.py
│   └── monitors.py
│
├── rfsn_controller/         # Main Controller
│   ├── controller.py        # 2600+ line repair loop
│   ├── controller_loop.py   # Serial execution with planner
│   ├── parallel.py          # Parallel patch evaluation
│   ├── evidence_pack.py     # Evidence collection & export
│   ├── setup_report.py      # Setup status reporting
│   ├── gates/               # Safety gates
│   ├── learning/            # Proposal-space learning
│   ├── qa/                  # QA/verification
│   ├── llm/                 # LLM integration & ensemble
│   └── buildpacks/          # Language support
│
├── tests/                   # Test Suite (102 tests)
│   ├── cgw/                 # CGW tests (18)
│   ├── test_phase2.py       # Phase 2 tests (37)
│   └── rfsn_controller/     # Controller tests
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

# Execution
--unsafe-host-exec       # Run locally (no Docker)
--steps 20               # Max repair steps
--parallel-patches       # Parallel patch evaluation
--ensemble-mode          # Multi-model ensemble
```

---

## 🔒 Security

- All code runs in isolated Docker containers (default)
- Optional `--unsafe-host-exec` for trusted local repos
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
