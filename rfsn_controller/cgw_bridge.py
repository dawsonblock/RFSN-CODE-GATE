"""CGW Bridge for RFSN Controller.

This module bridges the existing rfsn_controller logic to work through
the CGW serial decision architecture. It wraps the controller's
components as proposal generators and routes all decisions through
the thalamic gate.

Usage:
    from rfsn_controller.cgw_bridge import CGWControllerBridge
    from rfsn_controller.controller import ControllerConfig
    
    config = ControllerConfig(
        github_url="https://github.com/user/repo",
        test_cmd="pytest -q",
    )
    
    bridge = CGWControllerBridge(config)
    result = bridge.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from cgw_ssl_guard import SimpleEventBus, ThalamusGate, CGWRuntime
from cgw_ssl_guard.coding_agent import (
    AgentConfig,
    AgentResult,
    BlockingExecutor,
    CodingAction,
    CodingAgentRuntime,
    ExecutorConfig,
    PlannerProposalGenerator,
    ProposalContext,
    ProposalGenerator,
    SafetyProposalGenerator,
)
from cgw_ssl_guard.types import Candidate

if TYPE_CHECKING:
    from .controller import ControllerConfig
    from .sandbox import Sandbox
    from .planner_v2 import ControllerAdapter

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for the CGW-Controller bridge."""
    
    # Controller config passthrough
    github_url: str = ""
    test_cmd: str = "pytest -q"
    max_steps: int = 12
    
    # CGW-specific settings
    max_cycles: int = 100
    max_patches: int = 10
    enable_planner_v2: bool = True
    enable_memory: bool = True
    
    # Logging
    log_events: bool = True
    event_log_path: Optional[Path] = None


class ControllerProposalGenerator(ProposalGenerator):
    """Wraps existing controller logic as a proposal generator.
    
    This generator uses the controller's heuristics and pattern matching
    to propose actions based on the current state.
    """
    
    def __init__(
        self,
        sandbox: Optional["Sandbox"] = None,
        planner_adapter: Optional["ControllerAdapter"] = None,
    ):
        super().__init__("controller")
        self.sandbox = sandbox
        self.planner_adapter = planner_adapter
    
    def generate(self, context: ProposalContext) -> List[Candidate]:
        """Generate proposals based on controller logic."""
        candidates = []
        
        # Delegate to planner adapter if available
        if self.planner_adapter is not None:
            return self._generate_from_planner(context)
        
        # Fallback to basic heuristics
        return self._generate_heuristic(context)
    
    def _generate_from_planner(self, context: ProposalContext) -> List[Candidate]:
        """Generate proposals from PlannerV2 adapter."""
        if self.planner_adapter is None:
            return []
        
        # Check if we have an active plan
        if self.planner_adapter.is_complete() or self.planner_adapter.is_halted():
            # No more steps
            if self.planner_adapter.is_halted():
                return [self._make_candidate(
                    CodingAction.ABORT,
                    saliency=0.9,
                    urgency=0.8,
                    context={"reason": self.planner_adapter.get_halt_reason()},
                )]
            return [self._make_candidate(
                CodingAction.FINALIZE,
                saliency=1.0,
                context={"reason": "plan_complete"},
            )]
        
        # Get next task spec from planner
        # The planner's step types map to our coding actions
        summary = self.planner_adapter.get_summary()
        current_step = summary.get("current_step_index", 0)
        
        # Default to running tests if no specific action
        return [self._make_candidate(
            CodingAction.RUN_TESTS,
            saliency=0.7,
            urgency=0.5,
        )]
    
    def _generate_heuristic(self, context: ProposalContext) -> List[Candidate]:
        """Generate proposals using simple heuristics."""
        # Similar to PlannerProposalGenerator but with controller-specific logic
        candidates = []
        
        if context.last_action is None:
            # Start with tests
            candidates.append(self._make_candidate(
                CodingAction.RUN_TESTS,
                saliency=0.9,
                context={"reason": "initial"},
            ))
        elif context.last_action == CodingAction.RUN_TESTS:
            if context.tests_passing:
                candidates.append(self._make_candidate(
                    CodingAction.FINALIZE,
                    saliency=1.0,
                ))
            else:
                candidates.append(self._make_candidate(
                    CodingAction.ANALYZE_FAILURE,
                    saliency=0.85,
                ))
        elif context.last_action == CodingAction.ANALYZE_FAILURE:
            candidates.append(self._make_candidate(
                CodingAction.GENERATE_PATCH,
                saliency=0.8,
            ))
        elif context.last_action == CodingAction.GENERATE_PATCH:
            candidates.append(self._make_candidate(
                CodingAction.APPLY_PATCH,
                saliency=0.85,
            ))
        elif context.last_action == CodingAction.APPLY_PATCH:
            candidates.append(self._make_candidate(
                CodingAction.RUN_TESTS,
                saliency=0.9,
                context={"reason": "verify_patch"},
            ))
        
        return candidates


class CGWControllerBridge:
    """Bridge between the existing RFSN Controller and CGW runtime.
    
    This class adapts the existing controller infrastructure to work
    through the serial decision architecture. It:
    
    1. Wraps controller components as proposal generators
    2. Routes all decisions through the thalamic gate
    3. Uses the blocking executor for tool execution
    4. Emits events for replay and auditing
    """
    
    def __init__(
        self,
        config: Optional[BridgeConfig] = None,
        sandbox: Optional["Sandbox"] = None,
        planner_adapter: Optional["ControllerAdapter"] = None,
    ):
        self.config = config or BridgeConfig()
        self.sandbox = sandbox
        self.planner_adapter = planner_adapter
        
        # Event bus for all CGW events
        self.event_bus = SimpleEventBus()
        
        # Set up event logging if enabled
        if self.config.log_events:
            self._setup_event_logging()
        
        # Core CGW components
        self.gate = ThalamusGate(self.event_bus)
        self.cgw = CGWRuntime(self.event_bus)
        
        # Create executor with sandbox
        executor_config = ExecutorConfig(
            default_test_cmd=self.config.test_cmd,
        )
        self.executor = BlockingExecutor(
            sandbox=sandbox,
            config=executor_config,
        )
        
        # Create generators
        self.generators = self._create_generators()
        
        # Agent config
        self.agent_config = AgentConfig(
            goal=f"Fix tests for {self.config.github_url}",
            max_cycles=self.config.max_cycles,
            max_patches=self.config.max_patches,
        )
        
        # Runtime
        self.runtime: Optional[CodingAgentRuntime] = None
        
        # Event log
        self._event_log: List[Dict[str, Any]] = []
    
    def _setup_event_logging(self) -> None:
        """Set up event bus logging."""
        def log_event(event_name: str) -> Callable:
            def handler(payload: Any) -> None:
                self._event_log.append({
                    "event": event_name,
                    "payload": payload,
                })
            return handler
        
        self.event_bus.on("GATE_SELECTION", log_event("GATE_SELECTION"))
        self.event_bus.on("CGW_COMMIT", log_event("CGW_COMMIT"))
        self.event_bus.on("FORCED_INJECTION", log_event("FORCED_INJECTION"))
    
    def _create_generators(self) -> List[ProposalGenerator]:
        """Create the proposal generators for this bridge."""
        generators = [
            SafetyProposalGenerator(),
            ControllerProposalGenerator(
                sandbox=self.sandbox,
                planner_adapter=self.planner_adapter,
            ),
        ]
        
        # Add planner generator if enabled and adapter available
        if self.config.enable_planner_v2 and self.planner_adapter:
            generators.append(PlannerProposalGenerator(
                adapter=self.planner_adapter,
            ))
        
        return generators
    
    def run(self) -> Dict[str, Any]:
        """Run the controller through the CGW serial decision loop.
        
        Returns:
            Dictionary with success status, cycle count, and event log.
        """
        # Create runtime
        self.runtime = CodingAgentRuntime(
            config=self.agent_config,
            event_bus=self.event_bus,
            gate=self.gate,
            cgw=self.cgw,
            executor=self.executor,
            generators=self.generators,
        )
        
        # Run until done
        result = self.runtime.run_until_done()
        
        # Verify seriality
        seriality_ok = self.runtime.verify_seriality()
        
        return {
            "success": result.success,
            "final_action": result.final_action.value,
            "cycles_executed": result.cycles_executed,
            "total_time_ms": result.total_time_ms,
            "tests_passing": result.tests_passing,
            "patches_applied": result.patches_applied,
            "seriality_maintained": seriality_ok,
            "error": result.error,
            "event_log": self._event_log,
            "cycle_history": [
                {
                    "cycle_id": c.cycle_id,
                    "action": c.action.value,
                    "was_forced": c.was_forced,
                    "decision_time_ms": c.decision_time_ms,
                    "execution_time_ms": c.execution_time_ms,
                }
                for c in result.cycle_history
            ],
        }
    
    def inject_abort(self, reason: str = "user_abort") -> None:
        """Inject an abort signal that bypasses competition."""
        if self.runtime:
            self.runtime.inject_forced_signal(CodingAction.ABORT, reason)
    
    def get_event_log(self) -> List[Dict[str, Any]]:
        """Get the event log for replay/debugging."""
        return self._event_log.copy()
