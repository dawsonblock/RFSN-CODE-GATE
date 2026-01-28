#!/usr/bin/env python3
"""Demo: Run RFSN-style fix on a repository.

This demonstrates the core fix loop without the full controller.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def run_fix_demo(repo_dir: str):
    """Run the fix demo on a repository."""
    from rfsn_controller.llm.async_client import call_deepseek_async
    
    repo_path = Path(repo_dir)
    
    print("\n" + "=" * 70)
    print("🔧 RFSN Fix Demo - Autonomous Bug Fixing")
    print("=" * 70)
    print(f"\n📁 Repository: {repo_path}")
    
    # Step 1: Run tests to find failures
    print("\n📋 Step 1: Running tests to find failures...")
    result = subprocess.run(
        ["python3", "-m", "pytest", "-v", "--tb=short"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    
    if result.returncode == 0:
        print("   ✅ All tests pass! No fixes needed.")
        return
    
    print(f"   ❌ Tests failed!")
    
    # Extract test failures
    failures = []
    for line in result.stdout.split("\n"):
        if "FAILED" in line:
            failures.append(line.strip())
    
    for f in failures[:3]:
        print(f"      → {f}")
    
    # Step 2: Read the source files
    print("\n📖 Step 2: Reading source code...")
    source_files = {}
    for py_file in repo_path.glob("*.py"):
        if not py_file.name.startswith("test_"):
            source_files[py_file.name] = py_file.read_text()
            print(f"   → {py_file.name} ({len(py_file.read_text().split(chr(10)))} lines)")
    
    # Step 3: Ask LLM to fix
    print("\n🤖 Step 3: Asking LLM for fixes...")
    
    for filename, content in source_files.items():
        prompt = f"""Fix all bugs in this Python code. The tests are failing.

CRITICAL: Return ONLY the complete fixed Python code. NO JSON. NO tool requests. Just raw Python code.

Test failures:
{result.stdout[-1500:]}

Current buggy code ({filename}):
```python
{content}
```

Looking at the test failures:
- test_add fails: add(2, 3) returns -1 instead of 5 → the function does subtraction instead of addition
- test_factorial fails: factorial(5) causes recursion error → recursive call is wrong

Return the COMPLETE FIXED FILE. Start with "# Calculator module" and give me the entire corrected Python code:"""

        try:
            response = await call_deepseek_async(
                prompt,
                model="deepseek-chat",
                temperature=0.0,
            )
            
            # Parse response
            import json
            import re
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            fixed_code = None
            
            # Try to extract code from various response formats
            try:
                data = json.loads(response_content)
                if "diff" in data:
                    # Parse the diff
                    diff_content = data["diff"]
                    print(f"\n   📝 LLM provided patch for {filename}:")
                    
                    # Show some of the diff
                    for line in diff_content.split("\\n")[:6]:
                        if line.startswith("-") and not line.startswith("---"):
                            print(f"      \033[91m{line}\033[0m")
                        elif line.startswith("+") and not line.startswith("+++"):
                            print(f"      \033[92m{line}\033[0m")
                    
                    # Apply the diff
                    fixed_code = apply_diff_to_content(content, diff_content)
            except json.JSONDecodeError:
                pass
            
            # Check for raw Python code
            if not fixed_code:
                # Try to extract code from markdown blocks
                code_match = re.search(r'```python\n(.*?)```', response_content, re.DOTALL)
                if code_match:
                    fixed_code = code_match.group(1)
                elif response_content.strip().startswith(("def ", "# ", "import ", "class ")):
                    fixed_code = response_content
            
            if fixed_code and len(fixed_code.strip()) > 50:
                # Validate it looks like Python
                try:
                    compile(fixed_code, "<string>", "exec")
                    (repo_path / filename).write_text(fixed_code)
                    print(f"   ✅ Applied LLM fix to {filename}")
                except SyntaxError as e:
                    print(f"   ⚠️  Fix has syntax error: {e}")
            else:
                print(f"   ⚠️  Could not extract valid fix from response")
                    
        except Exception as e:
            print(f"   ❌ LLM error: {e}")
    
    # Step 4: Re-run tests
    print("\n🧪 Step 4: Re-running tests...")
    result = subprocess.run(
        ["python3", "-m", "pytest", "-v", "--tb=short"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    
    if result.returncode == 0:
        print("   ✅ All tests now pass!")
    else:
        # Count improvements
        new_failures = sum(1 for line in result.stdout.split("\n") if "FAILED" in line)
        old_failures = len(failures)
        if new_failures < old_failures:
            print(f"   📈 Progress: {old_failures} → {new_failures} failures")
        else:
            print(f"   ❌ Still {new_failures} failures")
    
    print("\n" + "=" * 70)


def apply_diff_to_content(original: str, diff: str) -> str:
    """Apply a unified diff to content and return the complete fixed file."""
    original_lines = original.split("\n")
    result_lines = []
    original_idx = 0
    
    # Parse the diff - split on actual newlines or escaped ones
    diff_lines = diff.replace("\\n", "\n").split("\n")
    
    in_hunk = False
    hunk_start = 0
    
    for diff_line in diff_lines:
        if diff_line.startswith("@@"):
            # Parse hunk header like @@ -5,7 +5,7 @@
            import re
            match = re.search(r'-(\d+)', diff_line)
            if match:
                hunk_start = int(match.group(1)) - 1  # 0-indexed
                # Copy any lines before this hunk
                while original_idx < hunk_start:
                    result_lines.append(original_lines[original_idx])
                    original_idx += 1
            in_hunk = True
            continue
        
        if not in_hunk:
            continue
            
        if diff_line.startswith("---") or diff_line.startswith("+++"):
            continue
        elif diff_line.startswith("-"):
            # Removed line - skip in original
            original_idx += 1
        elif diff_line.startswith("+"):
            # Added line
            result_lines.append(diff_line[1:])
        elif diff_line.startswith(" "):
            # Context line
            result_lines.append(diff_line[1:])
            original_idx += 1
        elif diff_line.strip() == "":
            # Empty line - could be context
            if original_idx < len(original_lines):
                result_lines.append(original_lines[original_idx])
                original_idx += 1
    
    # Add remaining lines from original
    while original_idx < len(original_lines):
        result_lines.append(original_lines[original_idx])
        original_idx += 1
    
    return "\n".join(result_lines) if result_lines else None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RFSN Fix Demo")
    parser.add_argument("repo", help="Path to repository to fix")
    args = parser.parse_args()
    
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("❌ Set DEEPSEEK_API_KEY environment variable")
        sys.exit(1)
    
    asyncio.run(run_fix_demo(args.repo))
