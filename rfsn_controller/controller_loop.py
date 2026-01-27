"""Controller Execution Loop with Planner Integration.

This module implements the SERIAL execution loop that integrates
the planner with the controller while maintaining safety guarantees.

EXECUTION MODEL:
1. Build observation packet (read-only)
2. Planner produces plan (JSON data)
3. PlanGate validates (HARD SAFETY)
4. For each step in topological order:
   - StepGate validates (per-step)
   - Execute step (only via controller tools)
   - Verify (blocking)
   - Record outcome
5. If step fails: planner may replan, but gate/budgets remain unchanged

KEY INVARIANTS:
- One mutation at a time (serial execution)
- Planner cannot execute (only proposes)
- Gate cannot be bypassed
- Learning cannot modify gates
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .gates.plan_gate import PlanGate, PlanGateConfig, PlanGateError, StepGateError
from .learning import LearnedStrategySelector

logger = logging.getLogger(__name__)


@dataclass
class ExecutionOutcome:
    """Outcome of a single step execution."""
    
    step_id: str
    success: bool
    elapsed_ms: int = 0
    error_message: Optional[str] = None
    diff: Optional[str] = None
    regression: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "success": self.success,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error_message,
            "regression": self.regression,
        }


@dataclass
class LoopResult:
    """Result of a full execution loop."""
    
    success: bool
    steps_executed: int
    steps_succeeded: int
    elapsed_ms: int
    outcomes: List[ExecutionOutcome] = field(default_factory=list)
    final_status: str = "unknown"
    replans: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "steps_executed": self.steps_executed,
            "steps_succeeded": self.steps_succeeded,
            "elapsed_ms": self.elapsed_ms,
            "final_status": self.final_status,
            "replans": self.replans,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


# Type for step executor function
StepExecutor = Callable[[Dict[str, Any]], ExecutionOutcome]
Verifier = Callable[[], bool]


class ControllerLoop:
    """Serial execution loop with planner integration.
    
    This loop maintains the "one mutation at a time" invariant
    while allowing the planner to guide step selection.
    
    Usage:
        loop = ControllerLoop(
            gate=PlanGate(config),
            executor=my_executor,
            verifier=my_verifier,
        )
        
        result = loop.run_with_planner(planner, observation)
    """
    
    def __init__(
        self,
        gate: Optional[PlanGate] = None,
        executor: Optional[StepExecutor] = None,
        verifier: Optional[Verifier] = None,
        learning: Optional[LearnedStrategySelector] = None,
        max_replans: int = 3,
        event_recorder: Optional[Callable[[str, Dict], None]] = None,
    ):
        """Initialize controller loop.
        
        Args:
            gate: PlanGate for validation. Creates default if None.
            executor: Function to execute a step.
            verifier: Function to verify after each step.
            learning: Optional learning selector for strategy suggestions.
            max_replans: Maximum replans on failure.
            event_recorder: Optional callback for event logging.
        """
        self.gate = gate or PlanGate()
        self.executor = executor or self._default_executor
        self.verifier = verifier or self._default_verifier
        self.learning = learning
        self.max_replans = max_replans
        self.event_recorder = event_recorder or self._default_recorder
        
        self._current_plan: Optional[Dict[str, Any]] = None
        self._completed_steps: List[str] = []
    
    def run_with_planner(
        self,
        planner: Any,  # PlannerV2 or compatible
        observation: Dict[str, Any],
    ) -> LoopResult:
        """Execute with planner-guided step selection.
        
        Args:
            planner: Planner instance with propose_plan method.
            observation: Read-only observation dict.
            
        Returns:
            LoopResult with execution summary.
        """
        start_time = time.monotonic()
        outcomes: List[ExecutionOutcome] = []
        replans = 0
        
        # Get initial plan from planner
        try:
            plan = planner.propose_plan(
                goal=observation.get("goal", ""),
                context=observation,
            )
            if hasattr(plan, "to_dict"):
                plan = plan.to_dict()
            elif not isinstance(plan, dict):
                plan = {"steps": [], "plan_id": "unknown"}
        except Exception as e:
            logger.error("Planner failed: %s", e)
            return LoopResult(
                success=False,
                steps_executed=0,
                steps_succeeded=0,
                elapsed_ms=int((time.monotonic() - start_time) * 1000),
                final_status=f"planner_error: {e}",
            )
        
        self._current_plan = plan
        self.event_recorder("PLAN_PROPOSED", {"plan_id": plan.get("plan_id")})
        
        # Validate plan
        try:
            self.gate.validate_plan(plan)
        except PlanGateError as e:
            logger.error("Plan validation failed: %s", e)
            self.event_recorder("PLAN_REJECTED", {"error": str(e)})
            return LoopResult(
                success=False,
                steps_executed=0,
                steps_succeeded=0,
                elapsed_ms=int((time.monotonic() - start_time) * 1000),
                final_status=f"plan_gate_error: {e}",
            )
        
        # Execute steps serially
        steps = plan.get("steps", [])
        steps_succeeded = 0
        
        for i, step in enumerate(steps):
            step_id = step.get("id", step.get("step_id", f"step_{i}"))
            
            # Per-step gate validation
            try:
                self.gate.validate_step(step)
            except StepGateError as e:
                logger.error("Step validation failed: %s", e)
                self.event_recorder("STEP_REJECTED", {"step_id": step_id, "error": str(e)})
                outcomes.append(ExecutionOutcome(
                    step_id=step_id,
                    success=False,
                    error_message=str(e),
                ))
                continue
            
            # Execute step
            step_start = time.monotonic()
            outcome = self.executor(step)
            outcome.elapsed_ms = int((time.monotonic() - step_start) * 1000)
            outcomes.append(outcome)
            
            # Verify (blocking)
            if not self.verifier():
                outcome.regression = True
                outcome.success = False
                outcome.error_message = "Verification failed"
            
            self.event_recorder("STEP_RESULT", outcome.to_dict())
            
            if outcome.success:
                steps_succeeded += 1
                self._completed_steps.append(step_id)
            else:
                # Replan on failure
                if replans < self.max_replans:
                    replans += 1
                    logger.info("Replanning after failure (replan %d/%d)", replans, self.max_replans)
                    self.event_recorder("PLAN_REPLAN", {"replan_count": replans, "failed_step": step_id})
                    
                    # Update observation with failure info
                    observation = dict(observation)
                    observation["last_failure"] = outcome.to_dict()
                    observation["completed_steps"] = self._completed_steps.copy()
                    
                    # Recurse with new plan
                    return self.run_with_planner(planner, observation)
                else:
                    self.event_recorder("PLAN_ABORTED", {"step_id": step_id, "replans": replans})
                    break
            
            # Update learning if available
            if self.learning:
                rec = self.learning.recommend(
                    failing_tests=observation.get("failing_tests"),
                    lint_errors=observation.get("lint_errors"),
                )
                self.learning.update(rec, success=outcome.success, regression=outcome.regression)
        
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        success = steps_succeeded == len(steps) and len(steps) > 0
        
        final_status = "success" if success else "partial"
        if replans >= self.max_replans:
            final_status = "max_replans_exceeded"
        
        self.event_recorder("PLAN_COMPLETE", {
            "success": success,
            "steps_executed": len(outcomes),
            "steps_succeeded": steps_succeeded,
        })
        
        return LoopResult(
            success=success,
            steps_executed=len(outcomes),
            steps_succeeded=steps_succeeded,
            elapsed_ms=elapsed_ms,
            outcomes=outcomes,
            final_status=final_status,
            replans=replans,
        )
    
    def run_plan(self, plan: Dict[str, Any]) -> LoopResult:
        """Execute a plan without planner (direct execution).
        
        Args:
            plan: Plan dictionary.
            
        Returns:
            LoopResult with execution summary.
        """
        start_time = time.monotonic()
        outcomes: List[ExecutionOutcome] = []
        
        # Validate plan
        try:
            self.gate.validate_plan(plan)
        except PlanGateError as e:
            return LoopResult(
                success=False,
                steps_executed=0,
                steps_succeeded=0,
                elapsed_ms=int((time.monotonic() - start_time) * 1000),
                final_status=f"plan_gate_error: {e}",
            )
        
        # Execute steps serially
        steps = plan.get("steps", [])
        steps_succeeded = 0
        
        for i, step in enumerate(steps):
            step_id = step.get("id", f"step_{i}")
            
            try:
                self.gate.validate_step(step)
            except StepGateError as e:
                outcomes.append(ExecutionOutcome(
                    step_id=step_id,
                    success=False,
                    error_message=str(e),
                ))
                continue
            
            outcome = self.executor(step)
            outcomes.append(outcome)
            
            if not self.verifier():
                outcome.regression = True
                outcome.success = False
            
            if outcome.success:
                steps_succeeded += 1
        
        return LoopResult(
            success=steps_succeeded == len(steps),
            steps_executed=len(outcomes),
            steps_succeeded=steps_succeeded,
            elapsed_ms=int((time.monotonic() - start_time) * 1000),
            outcomes=outcomes,
            final_status="success" if steps_succeeded == len(steps) else "partial",
        )
    
    def _default_executor(self, step: Dict[str, Any]) -> ExecutionOutcome:
        """Default no-op executor for testing."""
        return ExecutionOutcome(
            step_id=step.get("id", "unknown"),
            success=True,
        )
    
    def _default_verifier(self) -> bool:
        """Default verifier that always passes."""
        return True
    
    def _default_recorder(self, event_type: str, data: Dict) -> None:
        """Default event recorder that logs."""
        logger.info("Event: %s - %s", event_type, data)
