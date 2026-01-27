# rfsn_controller package

"""
RFSN Sandbox Controller - Autonomous software engineering agent.

Core Modules:
    - cli: command-line entry point
    - controller: the main RFSN controller loop
    - sandbox: utilities for managing disposable git sandboxes
    - verifier: test runner and result wrapper
    - parsers: helper functions for parsing test output
    - policy: heuristics for choosing repair intents and subgoals
    - prompt: helper for building model input strings
    - llm: LLM integrations (Gemini, DeepSeek, ensemble)
    - log: utility for writing JSONL logs

Optimization Modules:
    - docker_pool: warm container pool for faster Docker execution
    - semantic_cache: embedding-based cache for higher hit rates
    - prompt_compression: reduce token count for faster inference
    - streaming_validator: early termination for invalid responses
    - action_store: learn from past action outcomes
    - speculative_exec: predictive preloading
    - incremental_testing: run only affected tests

Planning Modules:
    - planner_v2: hierarchical task decomposition
    - cgw_bridge: Conscious Global Workspace integration
"""

# Convenience imports for common usage patterns
from .run_id import make_run_id

__all__ = [
    "make_run_id",
]

