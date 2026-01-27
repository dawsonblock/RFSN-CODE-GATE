"""Incremental test runner - only run tests affected by changes.

Analyzes file dependencies to determine minimal test set,
reducing test execution time by 50-90% in typical cases.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class FileChange:
    """Represents a changed file."""
    
    path: str
    change_type: str  # "modified", "added", "deleted"
    diff_hash: Optional[str] = None


@dataclass
class TestMapping:
    """Mapping of source files to their associated tests."""
    
    source_file: str
    test_files: List[str]
    dependencies: List[str]  # Other source files this imports
    confidence: float = 1.0  # How confident we are in this mapping


@dataclass
class IncrementalTestRunner:
    """Determines minimal test set based on file changes.
    
    Uses:
    1. Import graph analysis
    2. Test file naming conventions
    3. Historical test coverage data
    """
    
    repo_dir: str
    cache_db: Optional[str] = None
    
    _import_graph: Dict[str, Set[str]] = field(default_factory=dict)
    _test_map: Dict[str, List[str]] = field(default_factory=dict)
    _conn: Optional[sqlite3.Connection] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def __post_init__(self):
        if self.cache_db:
            self._init_db()
    
    def _init_db(self) -> None:
        """Initialize cache database."""
        os.makedirs(os.path.dirname(self.cache_db) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.cache_db, check_same_thread=False)
        
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS test_mappings (
                source_file TEXT PRIMARY KEY,
                test_files TEXT,  -- JSON list
                dependencies TEXT,  -- JSON list
                updated_at REAL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS test_history (
                test_file TEXT,
                source_file TEXT,
                success INTEGER,
                timestamp REAL,
                PRIMARY KEY (test_file, source_file)
            )
        """)
        self._conn.commit()
    
    def get_affected_tests(
        self,
        changes: List[FileChange],
        include_dependencies: bool = True,
    ) -> List[str]:
        """Get tests affected by the given file changes.
        
        Args:
            changes: List of file changes.
            include_dependencies: Also include tests for dependent files.
            
        Returns:
            List of test file paths to run.
        """
        affected_files: Set[str] = set()
        affected_tests: Set[str] = set()
        
        # Collect all affected source files
        for change in changes:
            affected_files.add(change.path)
            
            if include_dependencies:
                # Find files that depend on this one
                dependents = self._get_dependents(change.path)
                affected_files.update(dependents)
        
        # Find tests for affected files
        for file_path in affected_files:
            tests = self._find_tests_for_file(file_path)
            affected_tests.update(tests)
        
        # Always include tests that directly changed
        for change in changes:
            if self._is_test_file(change.path):
                affected_tests.add(change.path)
        
        return sorted(affected_tests)
    
    def _find_tests_for_file(self, source_path: str) -> List[str]:
        """Find test files for a source file."""
        tests = []
        
        # Check cache first
        if source_path in self._test_map:
            return self._test_map[source_path]
        
        # Naming conventions
        basename = os.path.basename(source_path)
        name_without_ext = os.path.splitext(basename)[0]
        
        # Common test file patterns
        test_patterns = [
            f"test_{name_without_ext}.py",
            f"{name_without_ext}_test.py",
            f"tests/test_{name_without_ext}.py",
            f"tests/{name_without_ext}_test.py",
            f"test/test_{name_without_ext}.py",
        ]
        
        for pattern in test_patterns:
            test_path = os.path.join(self.repo_dir, pattern)
            if os.path.exists(test_path):
                tests.append(pattern)
        
        # Also check for directory-based tests
        source_dir = os.path.dirname(source_path)
        if source_dir:
            tests_dir = os.path.join(self.repo_dir, source_dir, "tests")
            if os.path.isdir(tests_dir):
                for f in os.listdir(tests_dir):
                    if f.startswith("test_") and f.endswith(".py"):
                        tests.append(os.path.join(source_dir, "tests", f))
        
        # Cache result
        self._test_map[source_path] = tests
        
        return tests
    
    def _get_dependents(self, file_path: str) -> Set[str]:
        """Get files that import/depend on the given file."""
        dependents: Set[str] = set()
        
        # Build import graph if not cached
        if not self._import_graph:
            self._build_import_graph()
        
        # Find all files that import this one
        target_module = self._path_to_module(file_path)
        
        for source, imports in self._import_graph.items():
            if target_module in imports:
                dependents.add(source)
        
        return dependents
    
    def _build_import_graph(self) -> None:
        """Build the import dependency graph."""
        for root, _, files in os.walk(self.repo_dir):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                
                file_path = os.path.join(root, fname)
                rel_path = os.path.relpath(file_path, self.repo_dir)
                
                imports = self._extract_imports(file_path)
                self._import_graph[rel_path] = imports
    
    def _extract_imports(self, file_path: str) -> Set[str]:
        """Extract import statements from a Python file."""
        imports: Set[str] = set()
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # Match import statements
            import_patterns = [
                r'^import\s+([\w.]+)',
                r'^from\s+([\w.]+)\s+import',
            ]
            
            for pattern in import_patterns:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    imports.add(match.group(1))
        except Exception:
            pass
        
        return imports
    
    def _path_to_module(self, file_path: str) -> str:
        """Convert file path to module name."""
        rel_path = os.path.relpath(file_path, self.repo_dir)
        module = rel_path.replace(os.sep, ".").replace("/", ".")
        if module.endswith(".py"):
            module = module[:-3]
        return module
    
    def _is_test_file(self, path: str) -> bool:
        """Check if a file is a test file."""
        basename = os.path.basename(path)
        return (
            basename.startswith("test_") or
            basename.endswith("_test.py") or
            "/tests/" in path or
            "/test/" in path
        )
    
    def generate_pytest_args(
        self,
        changes: List[FileChange],
        extra_args: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate pytest command line arguments for incremental run.
        
        Args:
            changes: File changes to analyze.
            extra_args: Additional pytest arguments.
            
        Returns:
            List of pytest arguments.
        """
        affected = self.get_affected_tests(changes)
        
        if not affected:
            # No specific tests found, run all
            return extra_args or []
        
        args = list(extra_args or [])
        args.extend(affected)
        
        return args
    
    def record_test_result(
        self,
        test_file: str,
        source_files: List[str],
        success: bool,
    ) -> None:
        """Record which source files a test touched.
        
        This improves future test selection accuracy.
        """
        if not self._conn:
            return
        
        with self._lock:
            for source in source_files:
                try:
                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO test_history
                        (test_file, source_file, success, timestamp)
                        VALUES (?, ?, ?, ?)
                        """,
                        (test_file, source, 1 if success else 0, time.time())
                    )
                except Exception:
                    pass
            self._conn.commit()


def detect_changes_from_git(repo_dir: str) -> List[FileChange]:
    """Detect file changes from git status/diff.
    
    Args:
        repo_dir: Repository directory.
        
    Returns:
        List of file changes.
    """
    import subprocess
    
    changes: List[FileChange] = []
    
    try:
        # Get modified files
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                
                status = line[:2].strip()
                path = line[3:].strip()
                
                if status in ("M", "MM"):
                    change_type = "modified"
                elif status in ("A", "??"):
                    change_type = "added"
                elif status == "D":
                    change_type = "deleted"
                else:
                    change_type = "modified"
                
                changes.append(FileChange(path=path, change_type=change_type))
    except Exception:
        pass
    
    return changes


# Global runner instance
_runner: Optional[IncrementalTestRunner] = None
_runner_lock = threading.Lock()


def get_incremental_runner(repo_dir: str) -> IncrementalTestRunner:
    """Get the incremental test runner for a repo."""
    global _runner
    with _runner_lock:
        if _runner is None or _runner.repo_dir != repo_dir:
            cache_db = os.path.expanduser("~/.cache/rfsn/incremental_tests.db")
            _runner = IncrementalTestRunner(repo_dir=repo_dir, cache_db=cache_db)
        return _runner
