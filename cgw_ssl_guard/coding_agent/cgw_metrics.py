"""CGW Metrics and Dashboard.

This module provides Prometheus-compatible metrics for the CGW coding agent,
enabling monitoring of:
- Decision cycles
- Action execution
- LLM usage
- Sandbox operations

Also includes a simple HTML dashboard for real-time visualization.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class CGWMetricsCollector:
    """Collects and exposes metrics for the CGW coding agent.
    
    Tracks:
    - cgw_cycles_total: Total decision cycles
    - cgw_cycle_duration_seconds: Cycle timing histogram
    - cgw_actions_total: Actions by type
    - cgw_action_duration_seconds: Action execution time
    - cgw_llm_calls_total: LLM API calls
    - cgw_sandbox_operations_total: Sandbox operations
    - cgw_gate_selections: Gate selection outcomes
    
    Exposes metrics in Prometheus text format at /metrics endpoint.
    """
    
    def __init__(self):
        self._lock = Lock()
        
        # Counters
        self._cycles_total = 0
        self._actions_total: Dict[str, int] = defaultdict(int)
        self._llm_calls_total: Dict[str, int] = defaultdict(int)
        self._sandbox_ops_total: Dict[str, int] = defaultdict(int)
        self._gate_selections: Dict[str, int] = defaultdict(int)
        self._forced_overrides = 0
        
        # Gauges
        self._current_action: Optional[str] = None
        self._session_start_time: Optional[float] = None
        
        # Histograms (simplified - storing raw values)
        self._cycle_durations: List[float] = []
        self._action_durations: Dict[str, List[float]] = defaultdict(list)
        self._llm_latencies: List[float] = []
        
        # Recent events for dashboard
        self._recent_events: List[Dict[str, Any]] = []
        self._max_events = 100
    
    def start_session(self) -> None:
        """Mark the start of a coding session."""
        with self._lock:
            self._session_start_time = time.time()
            self._add_event("session_started", {})
    
    def record_cycle(self, duration_seconds: float, action: str) -> None:
        """Record a completed decision cycle.
        
        Args:
            duration_seconds: Time taken for the cycle.
            action: The action that was executed.
        """
        with self._lock:
            self._cycles_total += 1
            self._cycle_durations.append(duration_seconds)
            self._current_action = action
            self._add_event("cycle_completed", {
                "action": action,
                "duration_ms": duration_seconds * 1000,
            })
    
    def record_action(self, action: str, duration_seconds: float, success: bool) -> None:
        """Record an action execution.
        
        Args:
            action: Action type name.
            duration_seconds: Execution time.
            success: Whether action succeeded.
        """
        with self._lock:
            status = "success" if success else "failure"
            key = f"{action}:{status}"
            self._actions_total[key] += 1
            self._action_durations[action].append(duration_seconds)
            self._add_event("action_executed", {
                "action": action,
                "success": success,
                "duration_ms": duration_seconds * 1000,
            })
    
    def record_llm_call(
        self,
        provider: str,
        latency_seconds: float,
        success: bool,
    ) -> None:
        """Record an LLM API call.
        
        Args:
            provider: LLM provider name (deepseek, openai, gemini).
            latency_seconds: API latency.
            success: Whether call succeeded.
        """
        with self._lock:
            status = "success" if success else "failure"
            key = f"{provider}:{status}"
            self._llm_calls_total[key] += 1
            self._llm_latencies.append(latency_seconds)
    
    def record_sandbox_op(self, operation: str, success: bool) -> None:
        """Record a sandbox operation.
        
        Args:
            operation: Operation type (run, clone, apply_patch, etc.).
            success: Whether operation succeeded.
        """
        with self._lock:
            status = "success" if success else "failure"
            key = f"{operation}:{status}"
            self._sandbox_ops_total[key] += 1
    
    def record_gate_selection(
        self,
        winner_action: str,
        is_forced: bool,
        candidates_count: int,
    ) -> None:
        """Record a thalamic gate selection.
        
        Args:
            winner_action: The action that won.
            is_forced: Whether it was a forced override.
            candidates_count: Number of candidates competing.
        """
        with self._lock:
            self._gate_selections[winner_action] += 1
            if is_forced:
                self._forced_overrides += 1
            self._add_event("gate_selection", {
                "winner": winner_action,
                "is_forced": is_forced,
                "candidates": candidates_count,
            })
    
    def _add_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Add an event to the recent events list."""
        self._recent_events.append({
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        })
        # Trim to max events
        if len(self._recent_events) > self._max_events:
            self._recent_events = self._recent_events[-self._max_events:]
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus text format.
        
        Returns:
            Prometheus-formatted metrics string.
        """
        lines = []
        
        with self._lock:
            # Counters
            lines.append(f"# HELP cgw_cycles_total Total decision cycles")
            lines.append(f"# TYPE cgw_cycles_total counter")
            lines.append(f"cgw_cycles_total {self._cycles_total}")
            
            lines.append(f"# HELP cgw_actions_total Total actions by type")
            lines.append(f"# TYPE cgw_actions_total counter")
            for key, count in self._actions_total.items():
                action, status = key.split(":")
                lines.append(f'cgw_actions_total{{action="{action}",status="{status}"}} {count}')
            
            lines.append(f"# HELP cgw_llm_calls_total Total LLM API calls")
            lines.append(f"# TYPE cgw_llm_calls_total counter")
            for key, count in self._llm_calls_total.items():
                provider, status = key.split(":")
                lines.append(f'cgw_llm_calls_total{{provider="{provider}",status="{status}"}} {count}')
            
            lines.append(f"# HELP cgw_forced_overrides_total Total forced gate overrides")
            lines.append(f"# TYPE cgw_forced_overrides_total counter")
            lines.append(f"cgw_forced_overrides_total {self._forced_overrides}")
            
            # Gauges
            if self._session_start_time:
                uptime = time.time() - self._session_start_time
                lines.append(f"# HELP cgw_session_uptime_seconds Time since session started")
                lines.append(f"# TYPE cgw_session_uptime_seconds gauge")
                lines.append(f"cgw_session_uptime_seconds {uptime:.3f}")
            
            # Histograms (simplified summary stats)
            if self._cycle_durations:
                avg_cycle = sum(self._cycle_durations) / len(self._cycle_durations)
                lines.append(f"# HELP cgw_cycle_duration_seconds_avg Average cycle duration")
                lines.append(f"# TYPE cgw_cycle_duration_seconds_avg gauge")
                lines.append(f"cgw_cycle_duration_seconds_avg {avg_cycle:.6f}")
                lines.append(f"cgw_cycle_duration_seconds_count {len(self._cycle_durations)}")
        
        return "\n".join(lines)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for the HTML dashboard.
        
        Returns:
            Dictionary of dashboard data.
        """
        with self._lock:
            return {
                "cycles_total": self._cycles_total,
                "current_action": self._current_action,
                "actions": dict(self._actions_total),
                "llm_calls": dict(self._llm_calls_total),
                "forced_overrides": self._forced_overrides,
                "recent_events": list(self._recent_events[-20:]),
                "avg_cycle_time_ms": (
                    sum(self._cycle_durations) / len(self._cycle_durations) * 1000
                    if self._cycle_durations else 0
                ),
            }


# Module-level singleton
_metrics_collector: Optional[CGWMetricsCollector] = None


def get_metrics_collector() -> CGWMetricsCollector:
    """Get or create the metrics collector singleton."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = CGWMetricsCollector()
    return _metrics_collector


# --- HTML Dashboard ---

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CGW Coding Agent Dashboard</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border: #30363d;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --success: #3fb950;
            --warning: #d29922;
            --danger: #f85149;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }
        
        h1 {
            font-size: 24px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .status-badge {
            background: var(--success);
            color: #000;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        
        .card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }
        
        .card h3 {
            font-size: 12px;
            text-transform: uppercase;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }
        
        .metric {
            font-size: 36px;
            font-weight: 700;
            color: var(--accent);
        }
        
        .metric-unit {
            font-size: 14px;
            color: var(--text-secondary);
            margin-left: 4px;
        }
        
        .events-list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .event {
            padding: 12px;
            border-bottom: 1px solid var(--border);
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }
        
        .event:last-child { border-bottom: none; }
        
        .event-type {
            background: var(--bg-tertiary);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            flex-shrink: 0;
        }
        
        .event-data {
            font-size: 13px;
            color: var(--text-secondary);
            word-break: break-all;
        }
        
        .event-time {
            font-size: 11px;
            color: var(--text-secondary);
            margin-left: auto;
        }
        
        .action-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .action-chip {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 6px 12px;
            font-size: 12px;
        }
        
        .action-chip span {
            color: var(--accent);
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>
                🧠 CGW Coding Agent
                <span class="status-badge" id="status">ACTIVE</span>
            </h1>
            <div id="last-update" style="color: var(--text-secondary); font-size: 12px;"></div>
        </header>
        
        <div class="grid">
            <div class="card">
                <h3>Total Cycles</h3>
                <div class="metric" id="cycles">0</div>
            </div>
            
            <div class="card">
                <h3>Current Action</h3>
                <div class="metric" id="current-action" style="font-size: 24px;">IDLE</div>
            </div>
            
            <div class="card">
                <h3>Avg Cycle Time</h3>
                <div class="metric">
                    <span id="avg-cycle">0</span>
                    <span class="metric-unit">ms</span>
                </div>
            </div>
            
            <div class="card">
                <h3>Forced Overrides</h3>
                <div class="metric" id="forced">0</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>Actions Breakdown</h3>
                <div class="action-bar" id="actions"></div>
            </div>
            
            <div class="card">
                <h3>LLM Calls</h3>
                <div class="action-bar" id="llm-calls"></div>
            </div>
        </div>
        
        <div class="card">
            <h3>Recent Events</h3>
            <div class="events-list" id="events"></div>
        </div>
    </div>
    
    <script>
        async function fetchData() {
            try {
                const response = await fetch('/api/dashboard');
                const data = await response.json();
                
                document.getElementById('cycles').textContent = data.cycles_total;
                document.getElementById('current-action').textContent = data.current_action || 'IDLE';
                document.getElementById('avg-cycle').textContent = data.avg_cycle_time_ms.toFixed(0);
                document.getElementById('forced').textContent = data.forced_overrides;
                document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();
                
                // Actions
                const actionsEl = document.getElementById('actions');
                actionsEl.innerHTML = Object.entries(data.actions)
                    .map(([k, v]) => `<div class="action-chip">${k.split(':')[0]}: <span>${v}</span></div>`)
                    .join('');
                
                // LLM calls
                const llmEl = document.getElementById('llm-calls');
                llmEl.innerHTML = Object.entries(data.llm_calls)
                    .map(([k, v]) => `<div class="action-chip">${k}: <span>${v}</span></div>`)
                    .join('');
                
                // Events
                const eventsEl = document.getElementById('events');
                eventsEl.innerHTML = data.recent_events.reverse().map(e => `
                    <div class="event">
                        <span class="event-type">${e.type}</span>
                        <span class="event-data">${JSON.stringify(e.data)}</span>
                        <span class="event-time">${new Date(e.timestamp).toLocaleTimeString()}</span>
                    </div>
                `).join('');
                
            } catch (err) {
                console.error('Failed to fetch data:', err);
            }
        }
        
        fetchData();
        setInterval(fetchData, 2000);
    </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""
    
    def log_message(self, format: str, *args) -> None:
        """Suppress default logging."""
        pass
    
    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
            
        elif self.path == "/metrics":
            metrics = get_metrics_collector().get_prometheus_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(metrics.encode())
            
        elif self.path == "/api/dashboard":
            data = get_metrics_collector().get_dashboard_data()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
            
        else:
            self.send_response(404)
            self.end_headers()


def start_dashboard_server(port: int = 8765) -> HTTPServer:
    """Start the dashboard HTTP server in a background thread.
    
    Args:
        port: Port to listen on.
        
    Returns:
        The HTTPServer instance.
        
    Usage:
        server = start_dashboard_server(8765)
        # Dashboard available at http://localhost:8765
        # Metrics at http://localhost:8765/metrics
    """
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    logger.info(f"Dashboard server started at http://localhost:{port}")
    return server
