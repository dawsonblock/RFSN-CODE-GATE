"""Coding agent module for serial decision architecture.

This module provides a production-grade autonomous coding agent built
on the CGW (Conscious Global Workspace) / SSL guard architecture.

Key Features:
- Serial decision loop (exactly one decision per cycle)
- Thalamic gate arbitration with forced overrides
- Blocking execution (no tool overlap)
- Event emission for replay and auditing
- Seriality monitoring for invariant verification

Usage:
    from cgw_ssl_guard.coding_agent import CodingAgentRuntime, AgentConfig
    
    config = AgentConfig(goal="Fix failing tests", max_cycles=50)
    runtime = CodingAgentRuntime(config=config)
    result = runtime.run_until_done()
    
    if result.success:
        print(f"Fixed in {result.cycles_executed} cycles")
    else:
        print(f"Failed: {result.error}")

Architecture:
    Decision Layer (CGW + Gate)
    │
    ├── Proposal Generators → Candidates
    │   ├── SafetyProposalGenerator (ABORT on safety trigger)
    │   ├── PlannerProposalGenerator (next step from goal)
    │   ├── MemoryProposalGenerator (historical patterns)
    │   └── IdleProposalGenerator (fallback)
    │
    ├── Thalamic Gate → Single Winner Selection
    │   └── Forced queue checked first
    │
    ├── CGW Runtime → Atomic Commit
    │   └── One slot, atomic swap
    │
    └── Blocking Executor → Action Execution
        └── Runs tests, applies patches, etc.
        └── Returns results for next cycle
"""

from .action_types import (
    ActionCategory,
    ActionPayload,
    CodingAction,
    CycleResult,
    ExecutionResult,
    ACTION_CATEGORIES,
)

from .proposal_generators import (
    AnalyzerProposalGenerator,
    IdleProposalGenerator,
    MemoryProposalGenerator,
    PlannerProposalGenerator,
    ProposalContext,
    ProposalGenerator,
    SafetyProposalGenerator,
)

from .executor import (
    BlockingExecutor,
    ExecutorConfig,
    SandboxProtocol,
)

from .coding_agent_runtime import (
    AgentConfig,
    AgentResult,
    CodingAgentRuntime,
)

from .replay import (
    EventReplayEngine,
    ReplayCycle,
    ReplayEvent,
    SessionAnalysis,
)

from .llm_integration import (
    LLMAnalysisGenerator,
    LLMConfig,
    LLMDecisionAdvisor,
    LLMPatchGenerator,
)

__all__ = [
    # Action types
    "ActionCategory",
    "ActionPayload",
    "CodingAction",
    "CycleResult",
    "ExecutionResult",
    "ACTION_CATEGORIES",
    
    # Proposal generators
    "AnalyzerProposalGenerator",
    "IdleProposalGenerator",
    "MemoryProposalGenerator",
    "PlannerProposalGenerator",
    "ProposalContext",
    "ProposalGenerator",
    "SafetyProposalGenerator",
    
    # Executor
    "BlockingExecutor",
    "ExecutorConfig",
    "SandboxProtocol",
    
    # Runtime
    "AgentConfig",
    "AgentResult",
    "CodingAgentRuntime",
    
    # Replay
    "EventReplayEngine",
    "ReplayCycle",
    "ReplayEvent",
    "SessionAnalysis",
    
    # LLM Integration
    "LLMAnalysisGenerator",
    "LLMConfig",
    "LLMDecisionAdvisor",
    "LLMPatchGenerator",
]
