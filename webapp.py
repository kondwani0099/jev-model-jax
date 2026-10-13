#!/usr/bin/env python3
"""
JEV / UDM-200M Decision Studio Web Server.

Lightweight, high-performance HTTP server providing a modern UI
and REST API for testing the JAX Decision-Native Model.

Usage:
    python webapp.py                # Starts server on http://localhost:8080
    python webapp.py --port 8000    # Starts server on custom port
"""

import os
import sys
import json
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from infer_jax import JaxDecisionEngine
from udm_jax.distributed import detect_hardware

# Global decision engine instance
ENGINE = None
HARDWARE_INFO = None


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP Server for responsive concurrent decision calls."""
    daemon_threads = True


class DecisionStudioHandler(SimpleHTTPRequestHandler):
    """Handles static web UI assets and decision API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(PROJECT_ROOT, "web"), **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            return super().do_GET()

        elif self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = {
                "status": "healthy",
                "model_preset": "UDM-200M",
                "parameters": 201433483,
                "hardware": HARDWARE_INFO["target_name"],
                "backend": HARDWARE_INFO["backend"],
                "calibrated_temperature": getattr(ENGINE, "temperature", 1.5),
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))

        else:
            return super().do_GET()

    def do_POST(self):
        if self.path == "/api/decide":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                payload = json.loads(body.decode("utf-8"))
                domain = payload.get("domain", "finance")
                context = payload.get("context", {})
                decision_type = payload.get("decision_type", "classification")

                # Run inference via JAX engine
                result = ENGINE.decide(
                    context=context,
                    domain=domain,
                    decision_type=decision_type,
                )

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                err_resp = {"error": str(e)}
                self.wfile.write(json.dumps(err_resp).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {args[0]} {args[1]} -> {args[2]}\n")


def main():
    global ENGINE, HARDWARE_INFO
    parser = argparse.ArgumentParser(description="JEV-200M Decision Studio Web Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "udm_200m_jax"),
        help="Path to saved model checkpoint directory",
    )
    args = parser.parse_args()

    HARDWARE_INFO = detect_hardware()
    print("=" * 80)
    print("  JEV / UDM-200M DECISION STUDIO WEB APPLICATION")
    print("=" * 80)
    print(f"Hardware Platform:   {HARDWARE_INFO['target_name']} ({HARDWARE_INFO['backend'].upper()})")
    print(f"Loading Checkpoint:  {args.checkpoint_dir}")

    # Fallback to udm_125m_final if udm_200m_jax is not yet exported
    chk_dir = args.checkpoint_dir
    if not os.path.exists(os.path.join(chk_dir, "config.json")):
        alt_chk = os.path.join(PROJECT_ROOT, "checkpoints", "udm_125m_final")
        if os.path.exists(os.path.join(alt_chk, "config.json")):
            chk_dir = alt_chk

    ENGINE = JaxDecisionEngine(chk_dir)
    print(f"Engine Ready! Model loaded on {HARDWARE_INFO['target_name']}")
    print(f"Serving web application at: http://localhost:{args.port}\n")

    server_address = (args.host, args.port)
    httpd = ThreadedHTTPServer(server_address, DecisionStudioHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    main()
