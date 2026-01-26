<div align="center">

# 🚀 RFSN Controller

**Autonomous Code Repair Agent with Serial Decision Architecture**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)
[![CGW Architecture](https://img.shields.io/badge/CGW-Serial%20Decisions-purple.svg)](#cgw-mode)

*Fix bugs autonomously. One decision at a time.*

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

### ⚡ Multi-Model Ensemble

Active-active LLM failover.

- DeepSeek V3 primary
- Gemini 2.0 Flash fallback
- Thompson Sampling model selection
- Consensus voting on patches

</td>
</tr>
<tr>
<td width="50%">

### 📋 Planner v3.0

High-level goal decomposition.

- Failure classification
- Model arbitration learning
- Safety guardrails
- LLM-powered breakdown

</td>
<td width="50%">

### ⚖️ Adversarial QA

Every patch is guilty until proven innocent.

- Claim-based verification
- Evidence collection
- Accept/Reject/Escalate gates
- Regression firewall

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

# Run (classic mode)
python -m rfsn_controller.cli --repo https://github.com/user/repo --test "pytest -q"

# Run (CGW serial decision mode)
python -m rfsn_controller.cli --repo https://github.com/user/repo --cgw-mode

# Run (dedicated CGW CLI with event logging)
python -m rfsn_controller.cgw_cli --repo https://github.com/user/repo --save-events ./events.json
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

### Key Invariants

| Invariant | Enforcement |
|-----------|-------------|
| One decision/cycle | `SerialityMonitor` |
| Forced signals win | `inject_forced_signal()` |
| No tool overlap | `BlockingExecutor` |
| Replay support | Event emission |

### Replay Sessions

```python
from cgw_ssl_guard.coding_agent import EventReplayEngine

engine = EventReplayEngine.from_json("events.json")
analysis = engine.analyze()

print(analysis.summary())
# Session Analysis: SUCCESS
#   Cycles: 5
#   Seriality: OK
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

## 📁 Project Structure

```
├── cgw_ssl_guard/           # CGW/SSL Guard Core
│   ├── coding_agent/        # Serial Decision Coding Agent
│   │   ├── action_types.py
│   │   ├── proposal_generators.py
│   │   ├── executor.py
│   │   ├── coding_agent_runtime.py
│   │   ├── replay.py
│   │   └── llm_integration.py
│   ├── thalamic_gate.py
│   ├── event_bus.py
│   └── monitors.py
│
├── rfsn_controller/         # Main Controller
│   ├── controller.py        # 2600+ line repair loop
│   ├── planner_v2/          # Planner system
│   ├── qa/                   # QA/verification
│   ├── buildpacks/          # Language support
│   ├── cli.py               # Main CLI
│   ├── cgw_cli.py           # CGW CLI
│   └── cgw_bridge.py        # CGW integration
│
├── tests/                   # Test Suite
│   ├── cgw/                 # CGW tests
│   └── ...
│
└── docs/                    # Documentation
```

---

## 🧪 Testing

```bash
# Run CGW tests
pytest tests/cgw/ -v

# Run all tests
pytest tests/ -v

# Quick validation
python -c "from cgw_ssl_guard.coding_agent import CodingAgentRuntime; print('✓')"
```

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

# Parallel patches
--parallel-patches
--ensemble-mode

# Learning
--learning-db ./learning.db
--policy-mode bandit

# Planner
--planner-mode v2
```

---

## 🔒 Security

- All code runs in isolated Docker containers
- No host execution by default
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

</div>
