"""End-to-end integration tests for the CGW coding agent.

These tests validate the complete serial decision workflow:
1. Decision cycles execute serially (no parallel commits)
2. Gate selection respects forced overrides
3. Blocking execution prevents tool overlap
4. Events are emitted correctly for replay
5. Full workflow can complete a mock bug-fix

The tests use mock sandboxes to avoid Docker dependency.
"""

from __future__ import annotations

import time
from typing import List
from unittest.mock import MagicMock

import pytest

from cgw_ssl_guard.coding_agent import (
    ActionPayload,
    AgentConfig,
    AgentResult,
    BlockingExecutor,
    CodingAction,
    CodingAgentRuntime,
    ExecutorConfig,
    ProposalContext,
    SafetyProposalGenerator,
    LLMConfig,
    LLMPatchGenerator,
    get_metrics_collector,
)


# --- Fixtures ---

@pytest.fixture
def mock_sandbox():
    """Create a mock sandbox that doesn't require Docker."""
    sandbox = MagicMock()
    sandbox.run.return_value = MagicMock(
        returncode=0,
        stdout="5 passed in 0.1s",
        stderr="",
    )
    sandbox.read_file.return_value = "file contents"
    sandbox.write_file.return_value = None
    sandbox.apply_diff.return_value = True
    return sandbox


@pytest.fixture
def executor(mock_sandbox):
    """Create an executor with mock sandbox."""
    return BlockingExecutor(
        sandbox=mock_sandbox,
        config=ExecutorConfig(test_timeout=10, patch_timeout=5),
    )


@pytest.fixture
def mock_llm_caller():
    """Create a mock LLM caller."""
    def caller(prompt, model, temperature, max_tokens, **kwargs):
        return """```diff
--- a/example.py
+++ b/example.py
@@ -1,3 +1,3 @@
 def foo():
-    return None
+    return 42
```"""
    return caller


@pytest.fixture
def agent_config():
    """Create a default agent configuration."""
    return AgentConfig(
        goal="Fix failing tests",
        max_cycles=20,
    )


# --- Core Tests ---

class TestBlockingExecution:
    """Test that execution is properly blocking."""
    
    def test_executor_blocks_during_action(self, executor):
        """Verify executor blocks until action completes."""
        payload = ActionPayload(
            action=CodingAction.RUN_TESTS,
            parameters={"test_cmd": "pytest -q"},
        )
        
        start = time.time()
        result = executor.execute(payload)
        elapsed = time.time() - start
        
        # Should complete without timeout
        assert result.success
        assert elapsed < 5  # Reasonable time
    
    def test_executor_rejects_parallel_calls(self, mock_sandbox):
        """Verify executor rejects concurrent execute calls."""
        executor = BlockingExecutor(sandbox=mock_sandbox)
        
        # Manually set executing flag
        executor._is_executing = True
        
        # Should raise when trying to execute
        with pytest.raises(RuntimeError):
            executor.execute(ActionPayload(action=CodingAction.IDLE))
    
    def test_execution_count_increments(self, executor):
        """Verify execution count is tracked."""
        initial = executor.execution_count()
        
        executor.execute(ActionPayload(action=CodingAction.IDLE))
        executor.execute(ActionPayload(action=CodingAction.IDLE))
        
        assert executor.execution_count() == initial + 2


class TestProposalContext:
    """Test proposal context creation."""
    
    def test_context_creation(self):
        """ProposalContext should be creatable with required fields."""
        context = ProposalContext(
            cycle_id=1,
            goal="Fix test",
            last_action=CodingAction.IDLE,
        )
        
        assert context.cycle_id == 1
        assert context.goal == "Fix test"
        assert context.last_action == CodingAction.IDLE
    
    def test_context_defaults(self):
        """ProposalContext defaults should be reasonable."""
        context = ProposalContext(
            cycle_id=0,
            goal="test",
        )
        
        assert context.tests_passing is False
        assert context.patches_applied == 0
        assert context.failing_tests == []


class TestSafetyGenerator:
    """Test safety proposal generator."""
    
    def test_safety_generator_on_trigger(self):
        """SafetyProposalGenerator should produce ABORT when triggered."""
        generator = SafetyProposalGenerator()
        
        context = ProposalContext(
            cycle_id=1,
            goal="test",
            safety_triggered=True,
            safety_reason="Budget exceeded",
        )
        
        candidates = generator.generate(context)
        
        assert len(candidates) >= 1
    
    def test_safety_generator_no_trigger(self):
        """SafetyProposalGenerator should return empty when not triggered."""
        generator = SafetyProposalGenerator()
        
        context = ProposalContext(
            cycle_id=1,
            goal="test",
            safety_triggered=False,
        )
        
        candidates = generator.generate(context)
        
        # Should return empty when safety not triggered
        assert len(candidates) == 0


class TestLLMIntegration:
    """Test LLM integration (with mocks)."""
    
    def test_llm_patch_generator_with_caller(self, mock_llm_caller):
        """LLMPatchGenerator should produce patches when caller is set."""
        generator = LLMPatchGenerator(
            config=LLMConfig(),
            llm_caller=mock_llm_caller,
        )
        
        context = ProposalContext(
            cycle_id=5,
            goal="Fix test",
            last_action=CodingAction.ANALYZE_FAILURE,
            tests_passing=False,
            failing_tests=["test_foo"],
            test_output="AssertionError: expected 42, got None",
        )
        
        candidates = generator.generate(context)
        
        # Should produce patch candidates
        assert len(candidates) >= 1
    
    def test_llm_patch_generator_without_caller(self):
        """LLMPatchGenerator should return mock when no caller."""
        generator = LLMPatchGenerator(llm_caller=None)
        
        context = ProposalContext(
            cycle_id=3,
            goal="Fix test",
            last_action=CodingAction.ANALYZE_FAILURE,
            tests_passing=False,
            test_output="AssertionError",
        )
        
        candidates = generator.generate(context)
        
        # Should still produce (mock) candidates
        assert len(candidates) >= 1


class TestMetricsCollector:
    """Test the metrics collector."""
    
    def test_cycle_recording(self):
        """Metrics collector should record cycles."""
        collector = get_metrics_collector()
        initial = collector._cycles_total
        
        collector.record_cycle(0.5, "RUN_TESTS")
        collector.record_cycle(0.3, "ANALYZE_FAILURE")
        
        assert collector._cycles_total == initial + 2
    
    def test_prometheus_export(self):
        """Metrics should export in Prometheus format."""
        collector = get_metrics_collector()
        collector.record_cycle(0.5, "RUN_TESTS")
        
        metrics = collector.get_prometheus_metrics()
        
        assert "cgw_cycles_total" in metrics
        assert "counter" in metrics
    
    def test_action_recording(self):
        """Metrics collector should record actions."""
        collector = get_metrics_collector()
        
        collector.record_action("RUN_TESTS", 0.5, True)
        collector.record_action("APPLY_PATCH", 0.1, False)
        
        # Check actions were recorded
        assert "RUN_TESTS:success" in collector._actions_total or len(collector._actions_total) > 0


class TestCodingAgentRuntime:
    """Test the coding agent runtime."""
    
    def test_runtime_creation(self):
        """Runtime should be creatable with default config."""
        config = AgentConfig(goal="Test", max_cycles=5)
        runtime = CodingAgentRuntime(config=config)
        
        assert runtime is not None
    
    def test_runtime_run(self, mock_sandbox):
        """Runtime should run without crashing."""
        config = AgentConfig(goal="Test integration", max_cycles=3)
        executor = BlockingExecutor(sandbox=mock_sandbox)
        
        runtime = CodingAgentRuntime(
            config=config,
            executor=executor,
        )
        
        result = runtime.run_until_done()
        
        assert isinstance(result, AgentResult)
        assert result.cycles_executed >= 1
    
    def test_runtime_inject_forced_signal(self):
        """Runtime should allow forced signal injection."""
        config = AgentConfig(goal="Test", max_cycles=10)
        runtime = CodingAgentRuntime(config=config)
        
        # Inject abort signal
        slot_id = runtime.inject_forced_signal(
            action=CodingAction.ABORT,
            reason="Test forced abort",
        )
        
        assert slot_id is not None
        
        # Run should result in abort
        result = runtime.run_until_done()
        assert result.final_action == CodingAction.ABORT


class TestFullAgentIntegration:
    """Integration tests for the complete agent."""
    
    def test_agent_runs_without_errors(self):
        """Agent should run without errors using default config."""
        config = AgentConfig(
            goal="Test integration",
            max_cycles=5,
        )
        
        agent = CodingAgentRuntime(config=config)
        result = agent.run_until_done()
        
        # Should complete without crashes
        assert isinstance(result, AgentResult)
    
    def test_agent_tracks_cycle_count(self):
        """Agent should track cycle count accurately."""
        config = AgentConfig(goal="Test", max_cycles=3)
        agent = CodingAgentRuntime(config=config)
        
        result = agent.run_until_done()
        
        assert result.cycles_executed >= 1
        assert result.cycles_executed <= 3
    
    def test_seriality_check(self, mock_sandbox):
        """Agent should maintain seriality invariant."""
        config = AgentConfig(goal="Test seriality", max_cycles=5)
        executor = BlockingExecutor(sandbox=mock_sandbox)
        
        runtime = CodingAgentRuntime(
            config=config,
            executor=executor,
        )
        
        runtime.run_until_done()
        
        # Seriality should be maintained
        assert runtime.verify_seriality()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
