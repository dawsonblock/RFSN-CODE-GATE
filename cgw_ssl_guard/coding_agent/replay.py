"""Replay mechanism for CGW coding agent.

This module provides deterministic replay of CGW agent sessions from
event logs. It can be used to:

1. Debug failed runs by replaying decisions
2. Verify seriality invariants from logged events
3. Generate training data from successful runs
4. Test reproducibility of the decision architecture

Usage:
    from cgw_ssl_guard.coding_agent.replay import EventReplayEngine
    
    engine = EventReplayEngine.from_json("events.json")
    
    # Analyze the session
    analysis = engine.analyze()
    print(analysis["seriality_ok"])
    print(analysis["forced_signals"])
    
    # Replay step by step
    for cycle in engine.replay():
        print(f"Cycle {cycle.cycle_id}: {cycle.action}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .action_types import CodingAction

logger = logging.getLogger(__name__)


@dataclass
class ReplayEvent:
    """A single event from the CGW event log."""
    
    event_type: str
    payload: Dict[str, Any]
    
    @property
    def cycle_id(self) -> Optional[int]:
        return self.payload.get("cycle_id")
    
    @property
    def slot_id(self) -> Optional[str]:
        return self.payload.get("slot_id")
    
    @property
    def is_selection(self) -> bool:
        return self.event_type == "GATE_SELECTION"
    
    @property
    def is_commit(self) -> bool:
        return self.event_type == "CGW_COMMIT"
    
    @property
    def is_forced_injection(self) -> bool:
        return self.event_type == "FORCED_INJECTION"


@dataclass
class ReplayCycle:
    """A reconstructed decision cycle from events."""
    
    cycle_id: int
    action: CodingAction
    slot_id: str
    was_forced: bool
    losers: List[str]
    timestamp: float
    selection_reason: str = ""
    
    # Timing (if available)
    decision_time_ms: float = 0.0
    execution_time_ms: float = 0.0


@dataclass
class SessionAnalysis:
    """Analysis results from replaying a session."""
    
    total_cycles: int
    seriality_ok: bool
    seriality_violations: List[int]  # Cycle IDs with multiple commits
    
    forced_signals: int
    forced_cycle_ids: List[int]
    
    action_counts: Dict[str, int]
    final_action: CodingAction
    success: bool
    
    # Timing
    total_time_ms: float
    avg_cycle_time_ms: float
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        status = "SUCCESS" if self.success else "FAILURE"
        seriality = "OK" if self.seriality_ok else f"VIOLATED ({len(self.seriality_violations)} cycles)"
        return (
            f"Session Analysis: {status}\n"
            f"  Cycles: {self.total_cycles}\n"
            f"  Seriality: {seriality}\n"
            f"  Forced signals: {self.forced_signals}\n"
            f"  Final action: {self.final_action.value}\n"
            f"  Total time: {self.total_time_ms:.1f}ms"
        )


class EventReplayEngine:
    """Replay and analyze CGW coding agent sessions.
    
    This engine reconstructs decision cycles from event logs and
    provides analysis of the session's behavior.
    """
    
    def __init__(self, events: List[Dict[str, Any]]):
        """Initialize with raw event list.
        
        Args:
            events: List of event dictionaries with "event" and "payload" keys.
        """
        self.raw_events = events
        self.events = [
            ReplayEvent(event_type=e.get("event", ""), payload=e.get("payload", {}))
            for e in events
        ]
        self._cycles: Optional[List[ReplayCycle]] = None
    
    @classmethod
    def from_json(cls, path: Path | str) -> "EventReplayEngine":
        """Load events from a JSON file.
        
        Args:
            path: Path to the event log JSON file.
            
        Returns:
            EventReplayEngine instance.
        """
        path = Path(path)
        with open(path) as f:
            events = json.load(f)
        logger.info(f"Loaded {len(events)} events from {path}")
        return cls(events)
    
    @classmethod
    def from_result(cls, result: Dict[str, Any]) -> "EventReplayEngine":
        """Create from a CGW bridge result dictionary.
        
        Args:
            result: Result dictionary from CGWControllerBridge.run().
            
        Returns:
            EventReplayEngine instance.
        """
        events = result.get("event_log", [])
        return cls(events)
    
    def _build_cycles(self) -> List[ReplayCycle]:
        """Reconstruct cycles from events."""
        if self._cycles is not None:
            return self._cycles
        
        cycles = []
        commit_events = [e for e in self.events if e.is_commit]
        selection_events = {e.cycle_id: e for e in self.events if e.is_selection}
        
        for commit in commit_events:
            cycle_id = commit.cycle_id
            if cycle_id is None:
                continue
            
            # Get corresponding selection event
            selection = selection_events.get(cycle_id)
            
            # Determine action from slot_id
            slot_id = commit.slot_id or ""
            action = self._infer_action_from_slot(slot_id)
            
            cycle = ReplayCycle(
                cycle_id=cycle_id,
                action=action,
                slot_id=slot_id,
                was_forced=commit.payload.get("forced", False),
                losers=selection.payload.losers if selection else [],
                timestamp=commit.payload.get("timestamp", 0.0),
                selection_reason=commit.payload.get("reason", ""),
            )
            cycles.append(cycle)
        
        # Sort by cycle_id
        cycles.sort(key=lambda c: c.cycle_id)
        self._cycles = cycles
        return cycles
    
    def _infer_action_from_slot(self, slot_id: str) -> CodingAction:
        """Infer the action from the slot_id."""
        slot_upper = slot_id.upper()
        for action in CodingAction:
            if action.value in slot_upper:
                return action
        return CodingAction.IDLE
    
    def replay(self) -> Generator[ReplayCycle, None, None]:
        """Replay the session cycle by cycle.
        
        Yields:
            ReplayCycle for each decision cycle.
        """
        cycles = self._build_cycles()
        for cycle in cycles:
            logger.debug(f"Replay cycle {cycle.cycle_id}: {cycle.action.value}")
            yield cycle
    
    def analyze(self) -> SessionAnalysis:
        """Analyze the session for invariant violations and statistics.
        
        Returns:
            SessionAnalysis with detailed metrics.
        """
        cycles = self._build_cycles()
        
        # Check seriality
        commits_per_cycle: Dict[int, int] = {}
        for event in self.events:
            if event.is_commit and event.cycle_id is not None:
                commits_per_cycle[event.cycle_id] = commits_per_cycle.get(event.cycle_id, 0) + 1
        
        seriality_violations = [
            cycle_id for cycle_id, count in commits_per_cycle.items() if count > 1
        ]
        
        # Count forced signals
        forced_cycles = [c for c in cycles if c.was_forced]
        
        # Count actions
        action_counts: Dict[str, int] = {}
        for cycle in cycles:
            action_counts[cycle.action.value] = action_counts.get(cycle.action.value, 0) + 1
        
        # Determine final action and success
        final_action = cycles[-1].action if cycles else CodingAction.IDLE
        success = final_action == CodingAction.FINALIZE
        
        # Timing
        if len(cycles) >= 2:
            total_time_ms = (cycles[-1].timestamp - cycles[0].timestamp) * 1000
        else:
            total_time_ms = 0.0
        avg_cycle_time_ms = total_time_ms / len(cycles) if cycles else 0.0
        
        return SessionAnalysis(
            total_cycles=len(cycles),
            seriality_ok=len(seriality_violations) == 0,
            seriality_violations=seriality_violations,
            forced_signals=len(forced_cycles),
            forced_cycle_ids=[c.cycle_id for c in forced_cycles],
            action_counts=action_counts,
            final_action=final_action,
            success=success,
            total_time_ms=total_time_ms,
            avg_cycle_time_ms=avg_cycle_time_ms,
        )
    
    def verify_seriality(self) -> bool:
        """Quick check if seriality was maintained.
        
        Returns:
            True if no cycle had multiple commits.
        """
        analysis = self.analyze()
        return analysis.seriality_ok
    
    def get_action_sequence(self) -> List[CodingAction]:
        """Get the sequence of actions taken.
        
        Returns:
            List of CodingAction values in order.
        """
        return [c.action for c in self._build_cycles()]
    
    def find_forced_overrides(self) -> List[ReplayCycle]:
        """Find all cycles where a forced signal won.
        
        Returns:
            List of ReplayCycle instances with was_forced=True.
        """
        return [c for c in self._build_cycles() if c.was_forced]
    
    def export_training_data(self) -> List[Dict[str, Any]]:
        """Export session as training data for model fine-tuning.
        
        Returns:
            List of dictionaries suitable for training.
        """
        cycles = self._build_cycles()
        training_data = []
        
        for i, cycle in enumerate(cycles):
            # Get context from previous cycle
            prev_action = cycles[i-1].action if i > 0 else None
            
            training_data.append({
                "cycle_id": cycle.cycle_id,
                "context": {
                    "previous_action": prev_action.value if prev_action else None,
                },
                "decision": cycle.action.value,
                "was_forced": cycle.was_forced,
                "slot_id": cycle.slot_id,
            })
        
        return training_data
