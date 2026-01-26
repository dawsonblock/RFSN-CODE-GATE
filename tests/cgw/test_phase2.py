"""Phase 2 CGW Enhancement Tests.

Tests for all Phase 2 modules:
- CGWBandit: Strategy selection with UCB/Thompson Sampling
- CGWEventStore: Persistent event storage
- StreamingLLM: Async streaming with safety
- CGWActionMemory: Similarity boosting and regression firewall
- CGWDashboardServer: WebSocket real-time updates
"""

import json
import os
import sqlite3
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# CGWBandit Tests
# ============================================================================

class TestCGWBandit:
    """Tests for the CGW strategy bandit."""
    
    def test_bandit_creation(self):
        """Test bandit initializes with default arms."""
        from cgw_ssl_guard.coding_agent.cgw_bandit import CGWBandit, CGWBanditConfig
        
        bandit = CGWBandit()
        stats = bandit.get_stats()
        
        assert len(stats) == 8  # Default action types
        assert "RUN_TESTS" in stats
        assert "GENERATE_PATCH" in stats
    
    def test_bandit_update(self):
        """Test updating bandit with rewards."""
        from cgw_ssl_guard.coding_agent.cgw_bandit import CGWBandit
        
        bandit = CGWBandit()
        
        # Update with rewards
        bandit.update("RUN_TESTS", 1.0)
        bandit.update("RUN_TESTS", 1.0)
        bandit.update("GENERATE_PATCH", 0.0)
        
        stats = bandit.get_stats()
        
        # RUN_TESTS should have higher mean reward
        assert stats["RUN_TESTS"]["mean_reward"] > stats["GENERATE_PATCH"]["mean_reward"]
        assert stats["RUN_TESTS"]["pulls"] == 2
        assert stats["GENERATE_PATCH"]["pulls"] == 1
    
    def test_bandit_selection_thompson(self):
        """Test Thompson Sampling selection."""
        from cgw_ssl_guard.coding_agent.cgw_bandit import CGWBandit
        
        bandit = CGWBandit()
        
        # Update one arm heavily
        for _ in range(10):
            bandit.update("RUN_TESTS", 1.0)
        
        # Selection should favor RUN_TESTS
        selections = [bandit.select_action(method="thompson") for _ in range(20)]
        assert "RUN_TESTS" in selections
    
    def test_bandit_selection_ucb(self):
        """Test UCB selection."""
        from cgw_ssl_guard.coding_agent.cgw_bandit import CGWBandit
        
        bandit = CGWBandit()
        
        # Update one arm
        bandit.update("RUN_TESTS", 1.0)
        
        # UCB should explore unused arms first
        selected = bandit.select_action(method="ucb")
        # Should select an unexplored arm (infinite UCB)
        assert selected != "RUN_TESTS" or bandit._arms[selected].pulls == 0
    
    def test_bandit_saliency_boost(self):
        """Test saliency boosting."""
        from cgw_ssl_guard.coding_agent.cgw_bandit import CGWBandit
        
        bandit = CGWBandit()
        
        # Update with high success
        for _ in range(5):
            bandit.update("RUN_TESTS", 1.0)
        
        boost = bandit.get_saliency_boost("RUN_TESTS")
        
        # Should be boosted above 1.0
        assert boost > 1.0
        assert boost <= 1.5  # Max boost
    
    def test_bandit_avoidance_set(self):
        """Test avoidance set for failing actions."""
        from cgw_ssl_guard.coding_agent.cgw_bandit import CGWBandit
        
        bandit = CGWBandit()
        
        # Add many failures
        for _ in range(5):
            bandit.update("GENERATE_PATCH", 0.0)
        
        avoid = bandit.get_avoidance_set(threshold=0.3, min_pulls=3)
        
        assert "GENERATE_PATCH" in avoid
        assert "RUN_TESTS" not in avoid
    
    def test_bandit_persistence(self):
        """Test SQLite persistence."""
        from cgw_ssl_guard.coding_agent.cgw_bandit import CGWBandit, CGWBanditConfig
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "bandit.db")
            
            # Create and update bandit
            config = CGWBanditConfig(db_path=db_path)
            bandit1 = CGWBandit(config)
            bandit1.update("RUN_TESTS", 1.0)
            bandit1.update("RUN_TESTS", 1.0)
            bandit1.close()
            
            # Reload from database
            bandit2 = CGWBandit(CGWBanditConfig(db_path=db_path))
            stats = bandit2.get_stats()
            
            assert stats["RUN_TESTS"]["pulls"] == 2
            bandit2.close()
    
    def test_bandit_prometheus_metrics(self):
        """Test Prometheus metrics export."""
        from cgw_ssl_guard.coding_agent.cgw_bandit import CGWBandit
        
        bandit = CGWBandit()
        bandit.update("RUN_TESTS", 1.0)
        
        metrics = bandit.get_prometheus_metrics()
        
        assert "cgw_bandit_pulls_total" in metrics
        assert "cgw_bandit_mean_reward" in metrics
        assert 'action="RUN_TESTS"' in metrics
    
    def test_record_action_outcome_function(self):
        """Test recording action outcomes with fresh instance."""
        # Use a temp bandit to avoid singleton state issues
        from cgw_ssl_guard.coding_agent.cgw_bandit import CGWBandit, CGWBanditConfig
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_bandit.db")
            config = CGWBanditConfig(db_path=db_path)
            bandit = CGWBandit(config)
            
            bandit.update("RUN_TESTS", 1.0)
            bandit.update("RUN_TESTS", 0.5)
            
            stats = bandit.get_stats()
            assert stats["RUN_TESTS"]["pulls"] == 2
            
            bandit.close()



# ============================================================================
# EventStore Tests
# ============================================================================

class TestCGWEventStore:
    """Tests for the CGW event store."""
    
    def test_store_creation(self):
        """Test store initializes correctly."""
        from cgw_ssl_guard.coding_agent.event_store import CGWEventStore, EventStoreConfig
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "events.db")
            config = EventStoreConfig(db_path=db_path)
            store = CGWEventStore(config)
            
            stats = store.get_stats()
            assert stats["total_sessions"] == 0
            assert stats["total_events"] == 0
            store.close()
    
    def test_session_management(self):
        """Test session start and end."""
        from cgw_ssl_guard.coding_agent.event_store import CGWEventStore, EventStoreConfig
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "events.db")
            store = CGWEventStore(EventStoreConfig(db_path=db_path))
            
            store.start_session("test_session", goal="Fix tests")
            store.end_session("test_session", status="completed", total_cycles=10)
            
            sessions = store.get_recent_sessions(limit=10)
            assert len(sessions) == 1
            assert sessions[0]["session_id"] == "test_session"
            assert sessions[0]["status"] == "completed"
            store.close()
    
    def test_event_recording(self):
        """Test recording events."""
        from cgw_ssl_guard.coding_agent.event_store import CGWEventStore, EventStoreConfig
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "events.db")
            store = CGWEventStore(EventStoreConfig(db_path=db_path))
            
            store.start_session("test_session")
            
            event_id = store.record_event(
                session_id="test_session",
                event_type="CGW_COMMIT",
                cycle_id=1,
                payload={"action": "RUN_TESTS", "saliency": 0.9}
            )
            
            assert event_id == 1
            
            events = store.get_session_events("test_session")
            assert len(events) == 1
            assert events[0].event_type == "CGW_COMMIT"
            assert events[0].payload["action"] == "RUN_TESTS"
            store.close()
    
    def test_event_filtering(self):
        """Test event filtering by type and cycle."""
        from cgw_ssl_guard.coding_agent.event_store import CGWEventStore, EventStoreConfig
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "events.db")
            store = CGWEventStore(EventStoreConfig(db_path=db_path))
            
            store.start_session("test_session")
            store.record_event("test_session", "CGW_COMMIT", 1, {})
            store.record_event("test_session", "EXECUTION_COMPLETE", 1, {})
            store.record_event("test_session", "CGW_COMMIT", 2, {})
            
            # Filter by type
            commits = store.get_session_events("test_session", event_type="CGW_COMMIT")
            assert len(commits) == 2
            
            # Filter by cycle range
            cycle_1 = store.get_session_events("test_session", start_cycle=1, end_cycle=1)
            assert len(cycle_1) == 2
            store.close()
    
    def test_json_export(self):
        """Test JSON export."""
        from cgw_ssl_guard.coding_agent.event_store import CGWEventStore, EventStoreConfig
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "events.db")
            export_path = os.path.join(tmp, "export.json")
            store = CGWEventStore(EventStoreConfig(db_path=db_path))
            
            store.start_session("test_session")
            store.record_event("test_session", "CGW_COMMIT", 1, {"test": True})
            
            count = store.export_session_json("test_session", export_path)
            assert count == 1
            
            with open(export_path) as f:
                data = json.load(f)
            
            assert data["session_id"] == "test_session"
            assert len(data["events"]) == 1
            store.close()
    
    def test_event_iteration(self):
        """Test efficient event iteration."""
        from cgw_ssl_guard.coding_agent.event_store import CGWEventStore, EventStoreConfig
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "events.db")
            store = CGWEventStore(EventStoreConfig(db_path=db_path))
            
            store.start_session("test_session")
            for i in range(10):
                store.record_event("test_session", "CYCLE_END", i, {"cycle": i})
            
            events = list(store.iter_events("test_session", batch_size=3))
            assert len(events) == 10
            store.close()


# ============================================================================
# StreamingLLM Tests
# ============================================================================

class TestStreamingLLM:
    """Tests for streaming LLM functionality."""
    
    def test_streaming_metrics(self):
        """Test StreamingMetrics calculations."""
        from cgw_ssl_guard.coding_agent.streaming_llm import StreamingMetrics
        
        metrics = StreamingMetrics(
            start_time=1000.0,
            first_token_time=1000.1,
            end_time=1002.0,
            tokens_generated=100,
        )
        
        assert metrics.time_to_first_token_ms == pytest.approx(100.0, rel=0.01)
        assert metrics.total_time_ms == pytest.approx(2000.0, rel=0.01)
        assert metrics.tokens_per_second == pytest.approx(50.0, rel=0.01)
    
    def test_streaming_config(self):
        """Test StreamingConfig defaults."""
        from cgw_ssl_guard.coding_agent.streaming_llm import StreamingConfig
        
        config = StreamingConfig()
        
        assert config.max_tokens == 4096
        assert config.timeout == 120.0
        assert "rm -rf /" in config.safety_patterns
    
    def test_streaming_client_creation(self):
        """Test StreamingLLMClient initialization."""
        from cgw_ssl_guard.coding_agent.streaming_llm import (
            StreamingLLMClient, StreamingConfig
        )
        
        client = StreamingLLMClient(
            api_key="test_key",
            base_url="https://api.test.com",
            model="test-model",
        )
        
        assert client.api_key == "test_key"
        assert client.model == "test-model"
    
    def test_sync_wrapper_creation(self):
        """Test SyncStreamingWrapper initialization."""
        from cgw_ssl_guard.coding_agent.streaming_llm import SyncStreamingWrapper
        
        wrapper = SyncStreamingWrapper()
        assert wrapper._async_client is not None
    
    def test_safety_pattern_detection(self):
        """Test safety pattern detection."""
        from cgw_ssl_guard.coding_agent.streaming_llm import StreamingLLMClient
        
        client = StreamingLLMClient()
        
        assert client._check_safety("rm -rf /") == True
        assert client._check_safety("DROP TABLE users") == True
        assert client._check_safety("print('hello')") == False
    
    def test_create_streaming_caller(self):
        """Test streaming caller factory."""
        from cgw_ssl_guard.coding_agent.streaming_llm import create_streaming_caller
        
        caller = create_streaming_caller()
        assert callable(caller)


# ============================================================================
# ActionMemory Tests
# ============================================================================

class TestCGWActionMemory:
    """Tests for CGW action outcome memory."""
    
    def test_memory_creation(self):
        """Test memory initialization."""
        from cgw_ssl_guard.coding_agent.action_memory import (
            CGWActionMemory, CGWMemoryConfig
        )
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "memory.db")
            config = CGWMemoryConfig(db_path=db_path)
            memory = CGWActionMemory(config)
            
            stats = memory.get_stats()
            assert stats["available"] == True
            memory.close()
    
    def test_context_building(self):
        """Test building context signatures."""
        from cgw_ssl_guard.coding_agent.action_memory import (
            CGWActionMemory, CGWMemoryConfig
        )
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "memory.db")
            memory = CGWActionMemory(CGWMemoryConfig(db_path=db_path))
            
            context = memory.build_context(
                failure_class="assertion",
                repo_type="python",
                language="python",
                attempt_count=3,
            )
            
            assert context is not None
            assert context.failure_class == "assertion"
            assert context.attempt_bucket == 3
            memory.close()
    
    def test_outcome_recording(self):
        """Test recording action outcomes."""
        from cgw_ssl_guard.coding_agent.action_memory import (
            CGWActionMemory, CGWMemoryConfig
        )
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "memory.db")
            memory = CGWActionMemory(CGWMemoryConfig(db_path=db_path))
            memory.set_session("test_session")
            
            context = memory.build_context(failure_class="test", repo_type="python")
            
            success = memory.record_outcome(
                action_type="RUN_TESTS",
                action_key="test_key_123",
                outcome="success",
                context=context,
                exec_time_ms=1500,
            )
            
            assert success == True
            
            stats = memory.get_stats()
            assert stats["total_records"] == 1
            memory.close()
    
    def test_saliency_boost_without_history(self):
        """Test saliency boost with no history."""
        from cgw_ssl_guard.coding_agent.action_memory import (
            CGWActionMemory, CGWMemoryConfig
        )
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "memory.db")
            memory = CGWActionMemory(CGWMemoryConfig(db_path=db_path))
            
            boost = memory.get_saliency_boost(
                "RUN_TESTS",
                failure_class="test",
                repo_type="python",
            )
            
            # No history = no boost
            assert boost == 1.0
            memory.close()
    
    def test_regression_firewall_empty(self):
        """Test regression firewall with no history."""
        from cgw_ssl_guard.coding_agent.action_memory import (
            CGWActionMemory, CGWMemoryConfig
        )
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "memory.db")
            memory = CGWActionMemory(CGWMemoryConfig(db_path=db_path))
            
            is_blocked = memory.is_blocked(
                "RUN_TESTS",
                "some_key",
                failure_class="test",
                repo_type="python",
            )
            
            # No history = not blocked
            assert is_blocked == False
            memory.close()
    
    def test_prometheus_metrics(self):
        """Test Prometheus metrics export."""
        from cgw_ssl_guard.coding_agent.action_memory import (
            CGWActionMemory, CGWMemoryConfig
        )
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "memory.db")
            memory = CGWActionMemory(CGWMemoryConfig(db_path=db_path))
            
            metrics = memory.get_prometheus_metrics()
            
            assert "cgw_memory_records_total" in metrics
            memory.close()


# ============================================================================
# WebSocketDashboard Tests
# ============================================================================

class TestWebSocketDashboard:
    """Tests for WebSocket dashboard."""
    
    def test_token_cost_tracking(self):
        """Test token cost calculations."""
        from cgw_ssl_guard.coding_agent.websocket_dashboard import TokenCost
        
        cost = TokenCost()
        cost.add(1000, 500)  # 1000 prompt, 500 completion
        
        assert cost.prompt_tokens == 1000
        assert cost.completion_tokens == 500
        assert cost.total_tokens == 1500
        assert cost.cost_usd > 0
    
    def test_event_buffer(self):
        """Test circular event buffer."""
        from cgw_ssl_guard.coding_agent.websocket_dashboard import EventBuffer
        
        buffer = EventBuffer(max_size=5)
        
        for i in range(10):
            buffer.add({"id": i})
        
        events = buffer.get_all()
        assert len(events) == 5
        assert events[0]["id"] == 5  # Oldest kept
        assert events[-1]["id"] == 9  # Newest
    
    def test_event_buffer_since(self):
        """Test getting events since timestamp."""
        from cgw_ssl_guard.coding_agent.websocket_dashboard import EventBuffer
        
        buffer = EventBuffer()
        buffer.add({"timestamp": 100})
        buffer.add({"timestamp": 200})
        buffer.add({"timestamp": 300})
        
        events = buffer.get_since(150)
        assert len(events) == 2
    
    def test_dashboard_server_creation(self):
        """Test dashboard server initialization."""
        from cgw_ssl_guard.coding_agent.websocket_dashboard import (
            CGWDashboardServer, DashboardConfig
        )
        
        config = DashboardConfig(auto_open=False)
        server = CGWDashboardServer(config)
        
        assert server._running == False
        assert server.get_token_cost().total_tokens == 0
    
    def test_dashboard_event_emission(self):
        """Test event emission."""
        from cgw_ssl_guard.coding_agent.websocket_dashboard import (
            CGWDashboardServer, DashboardConfig
        )
        
        config = DashboardConfig(auto_open=False)
        server = CGWDashboardServer(config)
        
        server.emit_event({"event_type": "TEST", "data": "test"})
        
        events = server._events.get_all()
        assert len(events) == 1
        assert events[0]["event_type"] == "TEST"
    
    def test_dashboard_token_update(self):
        """Test token cost update."""
        from cgw_ssl_guard.coding_agent.websocket_dashboard import (
            CGWDashboardServer, DashboardConfig
        )
        
        config = DashboardConfig(auto_open=False)
        server = CGWDashboardServer(config)
        
        server.update_token_cost(100, 50)
        
        cost = server.get_token_cost()
        assert cost.prompt_tokens == 100
        assert cost.completion_tokens == 50
    
    def test_generate_dashboard_html(self):
        """Test HTML generation."""
        from cgw_ssl_guard.coding_agent.websocket_dashboard import generate_dashboard_html
        
        html = generate_dashboard_html()
        
        assert "CGW Real-Time Dashboard" in html
        assert "WebSocket" in html
        assert "tokenCost" in html.lower() or "token" in html.lower()


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase2Integration:
    """Integration tests for Phase 2 components working together."""
    
    def test_bandit_with_event_store(self):
        """Test bandit events are stored."""
        from cgw_ssl_guard.coding_agent.cgw_bandit import CGWBandit
        from cgw_ssl_guard.coding_agent.event_store import CGWEventStore, EventStoreConfig
        
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "events.db")
            store = CGWEventStore(EventStoreConfig(db_path=db_path))
            bandit = CGWBandit()
            
            store.start_session("test")
            
            # Simulate bandit action and record event
            action = bandit.select_action()
            store.record_event("test", "BANDIT_SELECT", 1, {"action": action})
            
            bandit.update(action, 1.0)
            store.record_event("test", "BANDIT_UPDATE", 1, {"action": action, "reward": 1.0})
            
            events = store.get_session_events("test")
            assert len(events) == 2
            store.close()
    
    def test_memory_with_bandit_boost(self):
        """Test memory and bandit working together."""
        from cgw_ssl_guard.coding_agent.cgw_bandit import CGWBandit
        from cgw_ssl_guard.coding_agent.action_memory import CGWActionMemory, CGWMemoryConfig
        
        with tempfile.TemporaryDirectory() as tmp:
            bandit = CGWBandit()
            memory = CGWActionMemory(CGWMemoryConfig(db_path=os.path.join(tmp, "mem.db")))
            
            # Get boosts from both
            bandit_boost = bandit.get_saliency_boost("RUN_TESTS")
            memory_boost = memory.get_saliency_boost("RUN_TESTS", failure_class="test")
            
            # Combine boosts
            combined = bandit_boost * memory_boost
            
            assert combined >= 0.64  # min_boost * 1.0
            memory.close()
    
    def test_all_imports_work(self):
        """Test all Phase 2 exports are available."""
        from cgw_ssl_guard.coding_agent import (
            # Bandit
            CGWBandit, CGWBanditConfig, get_cgw_bandit, record_action_outcome, BanditBoostMixin,
            # Event Store
            CGWEventStore, EventStoreConfig, get_event_store, StoredEvent, EventStoreSubscriber,
            # Streaming LLM
            StreamingLLMClient, StreamingConfig, StreamingMetrics, SyncStreamingWrapper, create_streaming_caller,
            # Action Memory
            CGWActionMemory, CGWMemoryConfig, get_action_memory, MemoryExecutorMixin,
            # Dashboard
            CGWDashboardServer, DashboardConfig, get_dashboard, DashboardEventSubscriber,
        )
        
        # All imports successful
        assert CGWBandit is not None
        assert CGWEventStore is not None
        assert StreamingLLMClient is not None
        assert CGWActionMemory is not None
        assert CGWDashboardServer is not None
