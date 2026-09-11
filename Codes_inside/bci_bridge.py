"""
bci_bridge.py  —  push detector events to the browser UI (no extra deps)
========================================================================
The detector runs in Python; the grid UI runs in the browser. This is the
one-way pipe between them: a tiny Server-Sent-Events (SSE) server on
http://localhost:8765/events that the browser reads with EventSource.

Model-independent: when you later swap in a retrained detector, nothing here
changes. Usage:

    bridge = BCIBridge(port=8765); bridge.start()
    ...
    bridge.push("SELECT")     # browser receives it, selects the focused tile

SSE (not websockets) is used deliberately: it needs only the standard library,
and the flow is one-way (Python -> browser), which is all a SELECT event needs.
"""

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class BCIBridge:
    def __init__(self, port=8765):
        self.port = port
        self._clients = []            # list of per-connection Queues
        self._lock = threading.Lock()
        self._server = None

    # ---- called by the detector ----
    def has_clients(self):
        return bool(self._clients)

    def push(self, event="SELECT", **data):
        if not self._clients:          # nobody watching -> do nothing (free)
            return
        msg = json.dumps({"event": event, **data})
        with self._lock:
            dead = []
            for q in self._clients:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._clients.remove(q)

    # ---- server plumbing ----
    def _make_handler(bridge):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence per-request logging
                pass

            def _cors(self):
                self.send_header("Access-Control-Allow-Origin", "*")

            def do_GET(self):
                if self.path.split("?")[0] != "/events":
                    self.send_response(404); self._cors(); self.end_headers(); return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self._cors()
                self.end_headers()
                q = queue.Queue(maxsize=100)
                with bridge._lock:
                    bridge._clients.append(q)
                try:
                    # initial hello so the browser knows it's connected
                    self.wfile.write(b"data: {\"event\": \"connected\"}\n\n")
                    self.wfile.flush()
                    while True:
                        try:
                            msg = q.get(timeout=15)
                            self.wfile.write(f"data: {msg}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")   # comment ping
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass   # browser tab closed/refreshed — normal, ignore quietly
                finally:
                    with bridge._lock:
                        if q in bridge._clients:
                            bridge._clients.remove(q)

            def handle_one_request(self):
                # swallow the same disconnect errors at the request level (Windows 10053)
                try:
                    super().handle_one_request()
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
                    self.close_connection = True
        return Handler

    def start(self):
        handler = self._make_handler()
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()
        print(f"[bridge] SSE on http://localhost:{self.port}/events")

    def stop(self):
        if self._server:
            self._server.shutdown()
