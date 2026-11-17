"""JSON-RPC 2.0 client for the BalatroBot mod (HTTP POST, stdlib only).

Every call has a timeout. A call that does not answer in time raises :class:`BridgeTimeout`
(the known BalatroBot hangs, see ``docs/design/bridge.md`` section 9, show up as timeouts);
a JSON-RPC error raises :class:`ApiError` with the BalatroBot error name
(``BAD_REQUEST``, ``INVALID_STATE``, ``NOT_ALLOWED``, ``INTERNAL_ERROR``).
"""

from __future__ import annotations

import http.client
import json
import time
from typing import Any

DEFAULT_URL = "http://127.0.0.1:12346"

ERROR_NAMES = {
    -32000: "INTERNAL_ERROR",
    -32001: "BAD_REQUEST",
    -32002: "INVALID_STATE",
    -32003: "NOT_ALLOWED",
}


class BridgeError(RuntimeError):
    """Base class of all bridge client errors."""


class BridgeTimeout(BridgeError):
    """The game did not answer within the timeout (likely a BalatroBot hang)."""


class ApiError(BridgeError):
    """The API answered with a JSON-RPC error."""

    def __init__(self, method: str, params: dict | None, code: int, message: str, name: str):
        super().__init__(f"{method}({params or {}}) -> {name}: {message}")
        self.method = method
        self.params = params
        self.code = code
        self.message = message
        self.name = name


class Client:
    """Thin JSON-RPC client. ``call(method, **params)`` returns the ``result`` object."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 60.0):
        rest = url.split("://", 1)[-1].rstrip("/")
        host, _, port = rest.partition(":")
        self.host = host or "127.0.0.1"
        self.port = int(port or 80)
        self.timeout = timeout
        self._id = 0
        self.calls = 0

    def call(self, method: str, timeout: float | None = None, **params: Any) -> Any:
        self._id += 1
        body = json.dumps(
            {"jsonrpc": "2.0", "method": method, "params": params, "id": self._id}
        ).encode()
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)
        try:
            conn.request("POST", "/", body=body, headers={"Content-Type": "application/json"})
            raw = conn.getresponse().read()
        except TimeoutError as e:
            raise BridgeTimeout(
                f"{method}({params}) timed out after {timeout or self.timeout}s"
            ) from e
        except (ConnectionError, OSError, http.client.HTTPException) as e:
            raise BridgeError(f"{method}({params}) connection failed: {e}") from e
        finally:
            conn.close()
        self.calls += 1
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as e:
            raise BridgeError(f"{method}: invalid JSON response ({raw[:200]!r})") from e
        if "error" in msg:
            err = msg["error"]
            code = int(err.get("code", -32000))
            name = (err.get("data") or {}).get("name") or ERROR_NAMES.get(code, "UNKNOWN")
            raise ApiError(method, params, code, str(err.get("message", "")), name)
        return msg.get("result")

    # -- conveniences ----------------------------------------------------------------------

    def health(self) -> bool:
        try:
            return self.call("health", timeout=5.0).get("status") == "ok"
        except BridgeError:
            return False

    def gamestate(self) -> dict:
        return self.call("gamestate")

    def wait_healthy(self, seconds: float = 120.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < seconds:
            if self.health():
                return True
            time.sleep(1.0)
        return False
