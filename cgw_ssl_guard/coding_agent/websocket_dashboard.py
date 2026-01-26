"""CGW WebSocket Dashboard.

Enhanced web dashboard with real-time WebSocket updates:
- Live event streaming from CGW event bus
- Session history browser
- Execution timeline visualization
- LLM token cost tracking

Uses a simple WebSocket server integrated with the existing dashboard.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import socket

logger = logging.getLogger(__name__)


@dataclass
class TokenCost:
    """Track token costs for LLM calls."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    
    # Pricing per 1M tokens (adjust as needed)
    INPUT_COST_PER_M: float = 0.14  # DeepSeek
    OUTPUT_COST_PER_M: float = 0.28
    
    def add(self, prompt: int, completion: int) -> None:
        """Add token counts."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.cost_usd += (
            prompt * self.INPUT_COST_PER_M / 1_000_000 +
            completion * self.OUTPUT_COST_PER_M / 1_000_000
        )
    
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DashboardConfig:
    """Configuration for the WebSocket dashboard."""
    
    # HTTP server port
    http_port: int = 8765
    
    # WebSocket server port
    ws_port: int = 8766
    
    # Maximum events to keep in memory
    max_events: int = 1000
    
    # Auto-open browser on start
    auto_open: bool = True
    
    # Custom static file directory
    static_dir: Optional[str] = None


class EventBuffer:
    """Thread-safe circular buffer for events."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def add(self, event: Dict[str, Any]) -> None:
        """Add an event to the buffer."""
        with self._lock:
            self._events.append(event)
            if len(self._events) > self.max_size:
                self._events = self._events[-self.max_size:]
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all events."""
        with self._lock:
            return list(self._events)
    
    def get_since(self, timestamp: float) -> List[Dict[str, Any]]:
        """Get events since a timestamp."""
        with self._lock:
            return [e for e in self._events if e.get("timestamp", 0) > timestamp]
    
    def clear(self) -> None:
        """Clear all events."""
        with self._lock:
            self._events.clear()


class WebSocketServer:
    """Simple WebSocket server for real-time updates.
    
    Note: This is a basic implementation using asyncio.
    For production, consider using websockets library.
    """
    
    def __init__(self, port: int = 8766):
        self.port = port
        self._clients: Set[Any] = set()
        self._server: Optional[Any] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        logger.info(f"WebSocket server starting on port {self.port}")
    
    def _run_server(self) -> None:
        """Run the server in a new event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
        finally:
            self._loop.close()
    
    async def _serve(self) -> None:
        """Serve WebSocket connections."""
        try:
            # Try to import websockets, fall back to basic socket if not available
            import websockets
            
            async def handler(websocket, path):
                self._clients.add(websocket)
                try:
                    async for message in websocket:
                        # Echo back for now
                        pass
                except Exception:
                    pass
                finally:
                    self._clients.discard(websocket)
            
            self._server = await websockets.serve(handler, "localhost", self.port)
            logger.info(f"WebSocket server running on ws://localhost:{self.port}")
            
            while self._running:
                await asyncio.sleep(0.1)
                
        except ImportError:
            # Fallback: basic socket server that just accepts connections
            logger.warning("websockets not installed, using fallback")
            await self._basic_serve()
    
    async def _basic_serve(self) -> None:
        """Basic fallback when websockets library not available."""
        while self._running:
            await asyncio.sleep(1)
    
    def broadcast(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected clients."""
        if not self._clients or not self._loop:
            return
        
        message = json.dumps(data)
        
        async def send_to_all():
            for client in list(self._clients):
                try:
                    await client.send(message)
                except Exception:
                    self._clients.discard(client)
        
        try:
            asyncio.run_coroutine_threadsafe(send_to_all(), self._loop)
        except Exception as e:
            logger.debug(f"Broadcast error: {e}")
    
    def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        if self._server:
            self._server.close()


def generate_dashboard_html() -> str:
    """Generate the enhanced dashboard HTML with WebSocket support."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CGW Real-Time Dashboard</title>
    <style>
        :root {
            --bg-dark: #0f0f23;
            --bg-card: #1a1a2e;
            --text: #e0e0e0;
            --accent: #00d4ff;
            --success: #00ff88;
            --warning: #ffaa00;
            --error: #ff4444;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(0,212,255,0.3);
        }
        .header h1 {
            font-size: 1.4rem;
            color: var(--accent);
        }
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--error);
        }
        .status-dot.connected { background: var(--success); }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .card {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h3 {
            color: var(--accent);
            font-size: 0.9rem;
            margin-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .metric-value {
            font-weight: bold;
            color: var(--success);
        }
        
        .timeline {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.1);
            max-height: 400px;
            overflow-y: auto;
        }
        .timeline h3 {
            color: var(--accent);
            font-size: 0.9rem;
            margin-bottom: 10px;
        }
        .event {
            display: flex;
            gap: 10px;
            padding: 8px;
            margin-bottom: 5px;
            background: rgba(0,0,0,0.3);
            border-radius: 4px;
            font-size: 0.8rem;
        }
        .event-time {
            color: #888;
            min-width: 80px;
        }
        .event-type {
            color: var(--accent);
            min-width: 120px;
        }
        .event-data { color: #ccc; }
        
        .cost-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #2d1f3d 100%);
        }
        .cost-value {
            font-size: 1.8rem;
            color: var(--success);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>CGW Real-Time Dashboard</h1>
        <div class="status">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">Disconnected</span>
        </div>
    </div>
    
    <div class="grid">
        <div class="card">
            <h3>Session Info</h3>
            <div class="metric">
                <span>Session ID</span>
                <span class="metric-value" id="sessionId">-</span>
            </div>
            <div class="metric">
                <span>Current Cycle</span>
                <span class="metric-value" id="cycleCount">0</span>
            </div>
            <div class="metric">
                <span>Status</span>
                <span class="metric-value" id="sessionStatus">idle</span>
            </div>
        </div>
        
        <div class="card cost-card">
            <h3>Token Costs</h3>
            <div class="cost-value">$<span id="totalCost">0.00</span></div>
            <div class="metric">
                <span>Prompt Tokens</span>
                <span id="promptTokens">0</span>
            </div>
            <div class="metric">
                <span>Completion Tokens</span>
                <span id="completionTokens">0</span>
            </div>
        </div>
        
        <div class="card">
            <h3>Execution Stats</h3>
            <div class="metric">
                <span>Total Actions</span>
                <span class="metric-value" id="totalActions">0</span>
            </div>
            <div class="metric">
                <span>Success Rate</span>
                <span class="metric-value" id="successRate">-</span>
            </div>
            <div class="metric">
                <span>Avg Exec Time</span>
                <span class="metric-value" id="avgExecTime">-</span>
            </div>
        </div>
    </div>
    
    <div class="timeline">
        <h3>Event Timeline</h3>
        <div id="events"></div>
    </div>

    <script>
        const wsPort = location.port === "8765" ? "8766" : "8766";
        let ws;
        let reconnectInterval;
        
        function connect() {
            ws = new WebSocket(`ws://localhost:${wsPort}`);
            
            ws.onopen = () => {
                document.getElementById('statusDot').classList.add('connected');
                document.getElementById('statusText').textContent = 'Connected';
                clearInterval(reconnectInterval);
            };
            
            ws.onclose = () => {
                document.getElementById('statusDot').classList.remove('connected');
                document.getElementById('statusText').textContent = 'Disconnected';
                reconnectInterval = setInterval(connect, 3000);
            };
            
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    handleEvent(data);
                } catch (e) {
                    console.error('Parse error:', e);
                }
            };
        }
        
        function handleEvent(data) {
            // Update metrics
            if (data.session_id) {
                document.getElementById('sessionId').textContent = 
                    data.session_id.slice(0, 8) + '...';
            }
            if (data.cycle_id !== undefined) {
                document.getElementById('cycleCount').textContent = data.cycle_id;
            }
            if (data.status) {
                document.getElementById('sessionStatus').textContent = data.status;
            }
            if (data.token_cost) {
                const cost = data.token_cost;
                document.getElementById('totalCost').textContent = 
                    cost.cost_usd.toFixed(4);
                document.getElementById('promptTokens').textContent = 
                    cost.prompt_tokens.toLocaleString();
                document.getElementById('completionTokens').textContent = 
                    cost.completion_tokens.toLocaleString();
            }
            
            // Add to timeline
            const eventsDiv = document.getElementById('events');
            const eventEl = document.createElement('div');
            eventEl.className = 'event';
            
            const time = new Date(data.timestamp * 1000).toLocaleTimeString();
            const eventType = data.event_type || data.type || 'UPDATE';
            const summary = JSON.stringify(data).slice(0, 100);
            
            eventEl.innerHTML = `
                <span class="event-time">${time}</span>
                <span class="event-type">${eventType}</span>
                <span class="event-data">${summary}</span>
            `;
            
            eventsDiv.insertBefore(eventEl, eventsDiv.firstChild);
            
            // Keep only last 50 events
            while (eventsDiv.children.length > 50) {
                eventsDiv.removeChild(eventsDiv.lastChild);
            }
        }
        
        // Start connection
        connect();
        
        // Fallback: poll API if WebSocket fails
        setInterval(async () => {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                try {
                    const resp = await fetch('/api/metrics');
                    if (resp.ok) {
                        const data = await resp.json();
                        handleEvent(data);
                    }
                } catch (e) {}
            }
        }, 5000);
    </script>
</body>
</html>'''


class CGWDashboardServer:
    """Enhanced dashboard server with WebSocket support.
    
    Usage:
        server = CGWDashboardServer()
        server.start()
        
        # Emit events
        server.emit_event({
            "event_type": "CGW_COMMIT",
            "cycle_id": 1,
            "action": "RUN_TESTS"
        })
    """
    
    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
    ):
        self.config = config or DashboardConfig()
        self._http_server: Optional[HTTPServer] = None
        self._ws_server: Optional[WebSocketServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._events = EventBuffer(self.config.max_events)
        self._token_cost = TokenCost()
        self._running = False
    
    def start(self) -> None:
        """Start both HTTP and WebSocket servers."""
        if self._running:
            return
        
        self._running = True
        
        # Start WebSocket server
        self._ws_server = WebSocketServer(port=self.config.ws_port)
        self._ws_server.start()
        
        # Start HTTP server
        self._start_http_server()
        
        if self.config.auto_open:
            time.sleep(0.5)
            webbrowser.open(f"http://localhost:{self.config.http_port}")
    
    def _start_http_server(self) -> None:
        """Start the HTTP server for the dashboard."""
        parent = self
        
        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress logs
            
            def do_GET(self):
                if self.path == "/" or self.path == "/index.html":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(generate_dashboard_html().encode())
                elif self.path == "/api/metrics":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    data = {
                        "token_cost": parent._token_cost.as_dict(),
                        "events": parent._events.get_all()[-10:],
                    }
                    self.wfile.write(json.dumps(data).encode())
                elif self.path == "/api/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(parent._events.get_all()).encode())
                else:
                    self.send_error(404)
        
        # Find available port
        port = self.config.http_port
        for _ in range(10):
            try:
                self._http_server = HTTPServer(("localhost", port), Handler)
                break
            except socket.error:
                port += 1
        
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            daemon=True
        )
        self._http_thread.start()
        logger.info(f"Dashboard server running at http://localhost:{port}")
    
    def emit_event(self, event: Dict[str, Any]) -> None:
        """Emit an event to all connected clients."""
        if "timestamp" not in event:
            event["timestamp"] = time.time()
        
        self._events.add(event)
        
        if self._ws_server:
            self._ws_server.broadcast(event)
    
    def update_token_cost(self, prompt: int, completion: int) -> None:
        """Update token cost tracking."""
        self._token_cost.add(prompt, completion)
        
        self.emit_event({
            "type": "TOKEN_UPDATE",
            "token_cost": self._token_cost.as_dict(),
        })
    
    def get_token_cost(self) -> TokenCost:
        """Get current token cost."""
        return self._token_cost
    
    def stop(self) -> None:
        """Stop all servers."""
        self._running = False
        
        if self._http_server:
            self._http_server.shutdown()
        
        if self._ws_server:
            self._ws_server.stop()


# === Event Bus Integration ===

class DashboardEventSubscriber:
    """Subscribe to event bus and broadcast to dashboard."""
    
    def __init__(
        self,
        dashboard: CGWDashboardServer,
        session_id: str = "",
    ):
        self.dashboard = dashboard
        self.session_id = session_id
    
    def subscribe(self, event_bus: Any) -> None:
        """Subscribe to CGW events."""
        event_types = [
            "GATE_SELECTION",
            "CGW_COMMIT",
            "CGW_CLEAR",
            "EXECUTION_START",
            "EXECUTION_COMPLETE",
            "CYCLE_START",
            "CYCLE_END",
            "LLM_STREAM_CHUNK",
            "LLM_STREAM_COMPLETE",
        ]
        
        for event_type in event_types:
            handler = self._make_handler(event_type)
            event_bus.on(event_type, handler)
    
    def _make_handler(self, event_type: str) -> Callable:
        def handler(payload: Any):
            event = {
                "event_type": event_type,
                "session_id": self.session_id,
                "timestamp": time.time(),
            }
            
            if isinstance(payload, dict):
                event.update(payload)
                
                # Track token costs if present
                if "prompt_tokens" in payload and "completion_tokens" in payload:
                    self.dashboard.update_token_cost(
                        payload["prompt_tokens"],
                        payload["completion_tokens"]
                    )
            
            self.dashboard.emit_event(event)
        
        return handler


# === Singleton Access ===

_dashboard_instance: Optional[CGWDashboardServer] = None


def get_dashboard(
    auto_start: bool = True,
) -> CGWDashboardServer:
    """Get or create the global dashboard instance."""
    global _dashboard_instance
    
    if _dashboard_instance is None:
        _dashboard_instance = CGWDashboardServer()
        if auto_start:
            _dashboard_instance.start()
    
    return _dashboard_instance


def reset_dashboard() -> None:
    """Reset the dashboard instance."""
    global _dashboard_instance
    if _dashboard_instance:
        _dashboard_instance.stop()
    _dashboard_instance = None
