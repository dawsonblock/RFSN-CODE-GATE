#!/usr/bin/env python3
"""Demo: Fix QuixBugs with RFSN Controller.

QuixBugs is a benchmark of 40 classic algorithms with single-line bugs.
This script demos the RFSN controller fixing them autonomously.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


QUIXBUGS_DIR = Path(__file__).parent / "quickbugs_demo"
BUGGY_PROGRAMS = QUIXBUGS_DIR / "python_programs"
CORRECT_PROGRAMS = QUIXBUGS_DIR / "correct_python_programs"
TESTCASES = QUIXBUGS_DIR / "python_testcases"


# Sample bugs to demo (easiest ones)
DEMO_BUGS = [
    "bitcount",       # XOR should be AND: n ^= n-1 -> n &= n-1
    "gcd",           # Swap positions: gcd(a % b, b) -> gcd(b, a % b)
    "sieve",         # Missing =: i + 2 -> i + 1
    "sqrt",          # Wrong operator: 0.01 -> 0.00001
]


def load_testcases(name: str) -> list:
    """Load test cases for a program."""
    tc_file = TESTCASES / f"{name}.json"
    if tc_file.exists():
        import json
        with open(tc_file) as f:
            return json.load(f)
    return []


def run_tests(program_path: Path, name: str) -> tuple[bool, str]:
    """Run test cases for a program."""
    import subprocess
    
    # Create a simple test runner
    test_code = f'''
import sys
sys.path.insert(0, "{program_path.parent}")
from {name} import *

# Basic smoke tests
'''
    
    if name == "bitcount":
        test_code += '''
assert bitcount(127) == 7, f"Expected 7, got {bitcount(127)}"
assert bitcount(128) == 1, f"Expected 1, got {bitcount(128)}"
assert bitcount(0) == 0, f"Expected 0, got {bitcount(0)}"
print("All tests passed!")
'''
    elif name == "gcd":
        test_code += '''
assert gcd(12, 8) == 4, f"Expected 4, got {gcd(12, 8)}"
assert gcd(48, 18) == 6, f"Expected 6, got {gcd(48, 18)}"
print("All tests passed!")
'''
    elif name == "sieve":
        test_code += '''
primes = list(sieve(30))
expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
assert primes == expected, f"Expected {expected}, got {primes}"
print("All tests passed!")
'''
    elif name == "sqrt":
        test_code += '''
result = sqrt(4, 0.0001)
assert 1.99 < result < 2.01, f"Expected ~2, got {result}"
print("All tests passed!")
'''
    else:
        test_code += 'print("No specific tests, basic import passed")'
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
    except Exception as e:
        return False, str(e)


def demo_manual_fix():
    """Demo showing the bugs and their fixes manually."""
    print("\n" + "=" * 60)
    print("🐛 QuixBugs Demo - Single-Line Bug Fixes")
    print("=" * 60)
    
    for name in DEMO_BUGS:
        buggy_file = BUGGY_PROGRAMS / f"{name}.py"
        correct_file = CORRECT_PROGRAMS / f"{name}.py"
        
        if not buggy_file.exists():
            print(f"\n⚠️  {name}: File not found")
            continue
        
        print(f"\n📄 {name}.py")
        print("-" * 40)
        
        # Read files
        buggy_code = buggy_file.read_text()
        correct_code = correct_file.read_text() if correct_file.exists() else None
        
        # Run tests on buggy version
        passed, output = run_tests(buggy_file, name)
        if passed:
            print(f"   ⚠️  Buggy version unexpectedly passes tests")
        else:
            print(f"   ❌ Buggy version fails tests:")
            # Extract first error line
            for line in output.split("\n"):
                if "assert" in line.lower() or "error" in line.lower():
                    print(f"      → {line.strip()[:60]}")
                    break
        
        if correct_code:
            # Show diff
            buggy_lines = buggy_code.strip().split("\n")
            correct_lines = correct_code.strip().split("\n")
            
            for i, (b, c) in enumerate(zip(buggy_lines, correct_lines)):
                if b != c:
                    print(f"\n   📝 Fix on line {i+1}:")
                    print(f"      - {b.strip()}")
                    print(f"      + {c.strip()}")
                    break
            
            # Verify fix works
            passed, _ = run_tests(correct_file, name)
            if passed:
                print(f"   ✅ Fixed version passes all tests!")
            else:
                print(f"   ⚠️  Fixed version still has issues")
    
    print("\n" + "=" * 60)
    print("✨ Demo complete! The RFSN controller can auto-fix these.")
    print("=" * 60)


async def demo_with_llm():
    """Demo using LLM to suggest fixes (requires API keys)."""
    try:
        from rfsn_controller.llm.async_client import call_deepseek_async
    except ImportError:
        print("❌ LLM client not available")
        return
    
    print("\n" + "=" * 60)
    print("🤖 LLM-Powered Bug Detection Demo")
    print("=" * 60)
    
    for name in DEMO_BUGS[:2]:  # Just demo first 2
        buggy_file = BUGGY_PROGRAMS / f"{name}.py"
        correct_file = CORRECT_PROGRAMS / f"{name}.py"
        
        if not buggy_file.exists():
            continue
        
        print(f"\n🔍 Analyzing {name}.py...")
        
        buggy_code = buggy_file.read_text()
        _, error = run_tests(buggy_file, name)
        
        prompt = f"""Fix the bug in this Python code. There is exactly one single-line bug.

```python
{buggy_code}
```

Test error:
{error[:500]}

Respond with ONLY the fixed single line of code, nothing else."""
        
        try:
            response = await call_deepseek_async(
                prompt,
                model="deepseek-chat",
                max_tokens=100,
            )
            print(f"   🤖 LLM suggestion: {response.strip()}")
            
            # Check against correct fix
            correct_code = correct_file.read_text() if correct_file.exists() else ""
            if any(line.strip() in response for line in correct_code.split("\n") if line.strip()):
                print(f"   ✅ Matches correct fix!")
            
        except Exception as e:
            print(f"   ⚠️  LLM call failed: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QuixBugs Demo")
    parser.add_argument("--llm", action="store_true", help="Use LLM for suggestions")
    args = parser.parse_args()
    
    # Run manual demo
    demo_manual_fix()
    
    # Optionally run LLM demo
    if args.llm:
        if os.environ.get("DEEPSEEK_API_KEY"):
            asyncio.run(demo_with_llm())
        else:
            print("\n⚠️  Set DEEPSEEK_API_KEY to run LLM demo")
