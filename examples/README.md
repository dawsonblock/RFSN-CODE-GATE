# Examples

This directory contains example repositories and demo scripts for testing the RFSN Controller.

## Test Repositories

| Directory | Description |
|-----------|-------------|
| `simple_bugs_repo/` | Single-bug calculator for quick testing |
| `hard_bugs_repo/` | 8 interrelated bugs across async/pipeline modules |
| `test_fix_repo/` | Basic test fixture repository |

## Demo Scripts

| Script | Description |
|--------|-------------|
| `demo_quixbugs.py` | QuixBugs benchmark demo |
| `demo_bugsinpy.py` | BugsInPy benchmark demo |
| `run_fix_demo.py` | Interactive fix demo runner |

## Usage

```bash
# Test simple bugs repo (1 bug)
python -m rfsn_controller.cli --repo examples/simple_bugs_repo --test "pytest test_calculator.py" --unsafe-host-exec

# Test hard bugs repo (8 bugs)
python -m rfsn_controller.cli --repo examples/hard_bugs_repo --test "pytest test_all.py" --unsafe-host-exec

# Run QuixBugs demo
python examples/demo_quixbugs.py
```
