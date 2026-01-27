# GitHub Copilot Instructions for RFSN Controller

## 🎯 Project Overview

RFSN Controller is an autonomous code repair agent with hierarchical planning and serial decision architecture. It uses a Conscious Global Workspace (CGW) architecture to enforce serial decision-making with safety guarantees.

**Key Principles:**
- Planner proposes, Gate authorizes, Controller executes
- One decision per cycle, no parallel chaos
- Safety-first architecture with hard gates
- Proposal-space learning only (learning cannot weaken gates)

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Package Manager:** pip, uv (alternative)
- **Build System:** hatchling
- **Testing Framework:** pytest
- **Linting:** ruff (replaces flake8/black)
- **Type Checking:** mypy
- **LLM Integration:** OpenAI API, Google GenAI
- **Web Framework:** FastAPI + uvicorn
- **Docker:** Containerized execution environment

## 🔨 Build & Test Commands

```bash
# Install dependencies
pip install -e .
pip install -e ".[dev]"      # Include dev dependencies
pip install -e ".[llm]"      # Include LLM dependencies

# Run all tests (102 tests total)
pytest tests/ -v

# Run specific test suites
pytest tests/rfsn_controller/test_phase4.py -v    # Phase 4 (hierarchical planner)
pytest tests/cgw/ -v                               # CGW tests
pytest tests/test_phase2.py -v                     # Phase 2 tests

# Linting with ruff
ruff check .
ruff format .

# Type checking
mypy rfsn_controller/
```

## 📏 Coding Conventions

### General Style
- **Line Length:** 120 characters (configured in pyproject.toml)
- **Target Version:** Python 3.11
- **Formatting:** Use ruff format (not black)
- **Import Sorting:** Managed by ruff (I001 rule)

### Python Specifics
- Use type hints for function signatures where meaningful
- Prefer dataclasses for structured data
- Use descriptive variable names (e.g., `plan_gate`, `strategy_selector`)
- Follow existing naming patterns in the codebase

### Architecture Patterns
- **Separation of Concerns:** Planner (proposes) → Gate (validates) → Executor (runs)
- **Event-Driven:** Use event bus for component communication
- **Immutable Plans:** Plans are JSON data only, never executable code
- **Safe Learning:** Learning only in proposal space, never weakens security gates

### Test Organization
- **Markers:** Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.security`
- **Fixtures:** Use fixtures from `tests/conftest.py`
- **Naming:** Test files must start with `test_`
- **Structure:** Group related tests in classes when appropriate

## 🏗️ Key Components

### Core Architecture
- `rfsn_controller/controller.py` - Main repair loop (2600+ lines)
- `rfsn_controller/controller_loop.py` - Serial execution with planner
- `rfsn_controller/gates/plan_gate.py` - Hard safety enforcement
- `rfsn_controller/learning/` - Proposal-space learning (fingerprint, bandit, quarantine)
- `cgw_ssl_guard/` - CGW/SSL Guard core (serial decision coding agent)

### Allowed Step Types
When working with plans, only these step types are permitted:
- **Read-only:** `search_repo`, `read_file`, `analyze_file`, `list_directory`, `grep_search`
- **Code modification:** `apply_patch`, `add_test`, `refactor_small`, `fix_import`, `fix_typing`
- **Verification:** `run_tests`, `run_lint`, `check_syntax`, `validate_types`
- **Coordination:** `wait`, `checkpoint`, `replan`

### Language Support (Buildpacks)
- `rfsn_controller/buildpacks/python_pack.py` - Python support
- `rfsn_controller/buildpacks/node_pack.py` - Node.js support
- `rfsn_controller/buildpacks/go_pack.py` - Go support
- `rfsn_controller/buildpacks/rust_pack.py` - Rust support
- `rfsn_controller/buildpacks/cpp_pack.py` - C/C++ support
- `rfsn_controller/buildpacks/java_pack.py` - Java support

## 🔒 Security Guidelines

### Critical Security Principles
1. **Never weaken security gates** - PlanGate validates every step before execution
2. **Shell injection detection** - All shell commands are validated
3. **Path validation** - All file operations are workspace-scoped only
4. **No host execution** - Code runs in isolated Docker containers by default
5. **Command allowlisting** - Only whitelisted commands are permitted
6. **APT package whitelisting** - Package installations are restricted

### Security Testing
- Always run security tests: `pytest tests/ -v -m security`
- Test shell scanner: `pytest tests/ -v -m scanner`
- Validate gate behavior before modifying gate logic

## ⚠️ What NOT to Modify

### Protected Files/Directories
- **Never disable or bypass:** `rfsn_controller/gates/plan_gate.py` security checks
- **Environment files:** Don't commit `.env` files or secrets
- **Build artifacts:** `__pycache__/`, `*.pyc`, `.pytest_cache/`, `dist/`, `build/`
- **Lock files:** Only regenerate `uv.lock` when dependencies actually change
- **Docker configs:** Modify `Dockerfile` and `docker-compose.yml` only when absolutely necessary

### Core Invariants (DO NOT BREAK)
1. **Planner never executes** - It produces JSON data only
2. **Gate has veto power** - Cannot be bypassed
3. **Learning cannot weaken gates** - Proposal space only
4. **Serial execution** - One mutation at a time
5. **No regressions** - Quarantine auto-blocks must remain active

## 📝 Documentation

When modifying features, update corresponding documentation:
- `docs/CGW_CODING_AGENT.md` - CGW architecture
- `docs/USAGE_GUIDE.md` - Usage guide
- `docs/FEATURE_MODE.md` - Feature engineering mode
- `docs/DOCKER_SANDBOX.md` - Docker sandbox setup
- `README.md` - Keep overview current (features, architecture, quick start)

## 🐛 Common Tasks

### Adding a New Step Type
1. Add to `DEFAULT_ALLOWED_STEP_TYPES` in `plan_gate.py`
2. Add validation logic in gate
3. Add tests in `tests/rfsn_controller/test_phase4.py`
4. Update documentation

### Adding a New Buildpack
1. Create new file in `rfsn_controller/buildpacks/`
2. Implement detection and test execution logic
3. Add tests in `tests/test_multi_language_support.py`
4. Update README language support table

### Modifying Learning Behavior
1. Changes must be in proposal space only (never execution)
2. Update learning components in `rfsn_controller/learning/`
3. Add tests ensuring gates cannot be weakened
4. Validate with Phase 4 tests

## 💡 Additional Notes

- This is a research/production hybrid codebase - balance safety with experimentation
- CGW mode enforces strict seriality - respect this constraint
- Phase 4 is complete - focus on stability and refinement
- The hierarchical planner uses Thompson Sampling for strategy selection
- Failure fingerprinting categorizes errors for better learning
- Quarantine lane prevents regressions automatically

## 🔄 CI/CD

GitHub Actions workflows:
- `.github/workflows/ci.yml` - Main CI pipeline
- `.github/workflows/rfsn_autofix.yml` - Automated fix workflow
- `.github/workflows/rfsn_open_pr.yml` - PR automation

All PRs must pass the CI pipeline before merging.
