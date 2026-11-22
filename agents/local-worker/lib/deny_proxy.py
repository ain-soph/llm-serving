#!/usr/bin/python3
"""Proxy that refuses every request and records it: HTTP(S)_PROXY for a local-worker run.

The sandbox already blocks outbound connections; this makes proxy-aware clients (the harness itself, curl,
pip, git over https, Python urllib ...) show *what* they tried to reach. Prints its port on the first stdout
line, then appends one JSON line per attempt to the log file given as argv[1].
"""
import json
import socketserver
import sys
import time

LOG = sys.argv[1]


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            line = self.rfile.readline(8192).decode("latin-1").strip()
            headers = []
            while True:
                h = self.rfile.readline(8192)
                if not h or h in (b"\r\n", b"\n"):
                    break
                headers.append(h.decode("latin-1").strip())
            host = next((h.split(":", 1)[1].strip() for h in headers if h.lower().startswith("host:")), "")
            parts = line.split(" ")
            rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "method": parts[0] if parts else "",
                   "target": parts[1] if len(parts) > 1 else "", "host": host}
            with open(LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
            body = b"blocked by local-worker: no network access\n"
            self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nContent-Length: "
                             + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
        except Exception:
            pass


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    srv = Server(("127.0.0.1", 0), Handler)
    print(srv.server_address[1], flush=True)
    srv.serve_forever()
