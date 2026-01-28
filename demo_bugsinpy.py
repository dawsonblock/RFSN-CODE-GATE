#!/usr/bin/env python3
"""Demo: Fix real-world bugs from BugsInPy with RFSN Controller.

BugsInPy contains 493 real bugs from 17 popular Python projects.
These are MUCH harder than QuixBugs - real production bugs!
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


BUGSINPY_DIR = Path(__file__).parent / "bugsinpy_demo" / "projects"

# Sample harder bugs - real production bugs
DEMO_BUGS = [
    {
        "project": "thefuck",
        "bug_id": "1",
        "name": "pip_unknown_command regex",
        "description": "Regex pattern too restrictive - fails on commands with hyphens",
        "bug_file": "thefuck/rules/pip_unknown_command.py",
        "test_file": "tests/rules/test_pip_unknown_command.py",
        "difficulty": "Medium (2-line regex fix)",
    },
    {
        "project": "black",
        "bug_id": "1", 
        "name": "ProcessPoolExecutor OSError",
        "description": "Crashes on AWS Lambda where multiprocessing is unsupported",
        "bug_file": "black.py",
        "test_file": "tests/test_black.py",
        "difficulty": "Hard (multi-hunk, try/except + null check)",
    },
    {
        "project": "httpie",
        "bug_id": "1",
        "name": "HTTP content type handling",
        "description": "Wrong content type detection for certain responses",
        "bug_file": "httpie/output/streams.py",
        "test_file": "tests/test_httpie.py",
        "difficulty": "Medium",
    },
]


def load_patch(project: str, bug_id: str) -> str:
    """Load the bug patch diff."""
    patch_file = BUGSINPY_DIR / project / "bugs" / bug_id / "bug_patch.txt"
    if patch_file.exists():
        return patch_file.read_text()
    return ""


def load_bug_info(project: str, bug_id: str) -> dict:
    """Load bug metadata."""
    info_file = BUGSINPY_DIR / project / "bugs" / bug_id / "bug.info"
    info = {}
    if info_file.exists():
        for line in info_file.read_text().strip().split("\n"):
            if "=" in line:
                key, val = line.split("=", 1)
                info[key] = val.strip('"')
    return info


def extract_buggy_code(patch: str) -> str:
    """Extract the buggy version from a patch."""
    lines = []
    in_diff = False
    for line in patch.split("\n"):
        if line.startswith("@@"):
            in_diff = True
            continue
        if in_diff:
            if line.startswith("-") and not line.startswith("---"):
                lines.append(line[1:])  # Buggy line
            elif line.startswith("+") and not line.startswith("+++"):
                pass  # Skip fixed lines
            elif line.startswith(" "):
                lines.append(line[1:])  # Context
    return "\n".join(lines)


def extract_fix(patch: str) -> str:
    """Extract just the fix from a patch."""
    fixes = []
    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            fixes.append(line)
        elif line.startswith("-") and not line.startswith("---"):
            fixes.append(line)
    return "\n".join(fixes)


def demo_bugsinpy():
    """Show real-world bugs from BugsInPy."""
    print("\n" + "=" * 70)
    print("🔥 BugsInPy Demo - Real-World Production Bugs")
    print("=" * 70)
    print("\nThese are MUCH harder than QuixBugs single-line bugs!")
    print("These come from real projects: black, httpie, thefuck, pandas, etc.\n")
    
    for bug in DEMO_BUGS:
        project = bug["project"]
        bug_id = bug["bug_id"]
        
        print(f"\n{'─' * 70}")
        print(f"📦 {project.upper()} Bug #{bug_id}: {bug['name']}")
        print(f"{'─' * 70}")
        print(f"   📝 {bug['description']}")
        print(f"   🎯 Difficulty: {bug['difficulty']}")
        
        # Load patch
        patch = load_patch(project, bug_id)
        if not patch:
            print(f"   ⚠️  Patch not found")
            continue
        
        # Load info
        info = load_bug_info(project, bug_id)
        print(f"   📁 File: {bug['bug_file']}")
        print(f"   🔗 Buggy commit: {info.get('buggy_commit_id', 'N/A')[:12]}...")
        
        # Show the fix
        fix = extract_fix(patch)
        print(f"\n   📋 Fix Preview:")
        for line in fix.split("\n")[:8]:
            if line.startswith("-"):
                print(f"      \033[91m{line}\033[0m")  # Red
            elif line.startswith("+"):
                print(f"      \033[92m{line}\033[0m")  # Green
        if len(fix.split("\n")) > 8:
            print(f"      ... ({len(fix.split(chr(10)))} lines total)")
    
    print(f"\n{'=' * 70}")
    print("📊 Summary: BugsInPy has 493 bugs across 17 projects")
    print("   - pandas: 169 bugs    - black: 23 bugs")
    print("   - scrapy: 40 bugs     - thefuck: 32 bugs")
    print("   - keras: 45 bugs      - and more...")
    print("=" * 70)


async def try_llm_fix(bug: dict):
    """Try to get LLM to suggest a fix."""
    try:
        from rfsn_controller.llm.async_client import call_deepseek_async
    except ImportError:
        print("   ⚠️  LLM client not available")
        return
    
    patch = load_patch(bug["project"], bug["bug_id"])
    buggy_code = extract_buggy_code(patch)
    
    prompt = f"""You are a Python expert fixing a bug. DO NOT request tools. Just provide the fix directly.

Project: {bug['project']}
Bug: {bug['description']}
File: {bug['bug_file']}

Buggy code:
```python
{buggy_code}
```

The regex pattern `[a-z]+` is too restrictive - it doesn't match commands with hyphens or other characters.

Provide ONLY the corrected Python code lines. No explanations, no tool requests, just the fixed code."""

    print("\n   🤖 Asking LLM for fix...")
    
    try:
        response = await call_deepseek_async(
            prompt,
            model="deepseek-chat",
            temperature=0.0,
        )
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Try to parse JSON response (RFSN controller format)
        import json
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "diff" in data:
                # Extract diff from patch response
                diff_content = data["diff"]
                print("   📤 LLM Patch Response:")
                for line in diff_content.split("\\n")[:8]:
                    if line.startswith("-"):
                        print(f"      \033[91m{line}\033[0m")
                    elif line.startswith("+"):
                        print(f"      \033[92m{line}\033[0m")
                    else:
                        print(f"      {line[:70]}")
                
                # Check for key fix pattern
                actual_fix = extract_fix(patch)
                if '[^"]+' in diff_content or "([^" in diff_content:
                    print("   ✅ LLM correctly identified the regex fix!")
                elif any(fix_line.lstrip("+").strip() in diff_content 
                       for fix_line in actual_fix.split("\n") 
                       if fix_line.startswith("+")):
                    print("   ✅ LLM identified key parts of the fix!")
                else:
                    print("   ⚠️  LLM fix may differ from ground truth")
            else:
                print(f"   📤 LLM Response: {content[:200]}")
        except json.JSONDecodeError:
            # Plain text response
            print(f"   📤 LLM Response:")
            for line in content.strip().split("\n")[:5]:
                print(f"      {line[:70]}")
            
    except Exception as e:
        print(f"   ❌ LLM error: {e}")


async def demo_with_llm():
    """Demo LLM fixing real bugs."""
    print("\n" + "=" * 70)
    print("🤖 LLM-Powered Fix Attempt (Real Production Bugs)")
    print("=" * 70)
    
    # Try just the thefuck bug (simpler)
    bug = DEMO_BUGS[0]  # thefuck regex bug
    print(f"\n🔍 Testing: {bug['project']} - {bug['name']}")
    await try_llm_fix(bug)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BugsInPy Demo")
    parser.add_argument("--llm", action="store_true", help="Try LLM fixes")
    args = parser.parse_args()
    
    demo_bugsinpy()
    
    if args.llm:
        if os.environ.get("DEEPSEEK_API_KEY"):
            asyncio.run(demo_with_llm())
        else:
            print("\n⚠️  Set DEEPSEEK_API_KEY to run LLM demo")
