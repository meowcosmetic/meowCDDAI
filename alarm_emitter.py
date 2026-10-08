"""
Asynchronous Fail-safe Alarm Emitter for Python FastAPI Services
Sends error payloads to Local Agent Bridge at http://localhost:8765/api/alarms/ingest
"""

import json
import threading
import traceback
import urllib.request

BRIDGE_URL = "http://localhost:8765/api/alarms/ingest"

def emit_alarm_async(service: str, endpoint: str, exc: Exception):
    def _send():
        try:
            tb = traceback.extract_tb(exc.__traceback__)
            location = "Unknown:0"
            suspected_file = ""
            if tb:
                last_frame = tb[-1]
                location = f"{last_frame.filename}:{last_frame.lineno}"
                suspected_file = last_frame.filename

            payload = {
                "service": service,
                "endpoint": endpoint,
                "status": 500,
                "exception": exc.__class__.__name__,
                "message": str(exc) or "Internal Server Error",
                "location": location,
                "suspectedFile": suspected_file,
            }

            req = urllib.request.Request(
                BRIDGE_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                pass
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()
