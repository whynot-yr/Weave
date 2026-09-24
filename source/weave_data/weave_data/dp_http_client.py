"""Single-session synchronous HTTP client; no automatic inference retries."""

import math
import urllib.error
import urllib.parse
import urllib.request
import uuid

from .dp_protocol import MAX_BODY_BYTES, WIRE_VERSION, array, encode_rgb, json_bytes, read_json, validate_health


class DPHttpClient:
    def __init__(self, url, timeout=30.0):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("V1 supports loopback HTTP only (no remote/public service)")
        if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("Expected a plain loopback base URL")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive")
        self.url, self.timeout = url.rstrip("/"), timeout
        self.session_id = str(uuid.uuid4())
        self.episode_id = -1
        self.request_id = 0
        self.last_control_step = -1
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.health = self._request("/health")
        self.protocol = validate_health(self.health)

    def _request(self, path, payload=None):
        request = urllib.request.Request(
            self.url + path,
            data=None if payload is None else json_bytes(payload),
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result = read_json(response.read(MAX_BODY_BYTES + 1))
        except urllib.error.HTTPError as exc:
            detail = exc.read(MAX_BODY_BYTES).decode("utf-8", errors="replace")
            raise RuntimeError(f"DP HTTP {exc.code} at {path}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"DP HTTP request failed: {path}: {exc}") from exc
        if result.get("protocol_version") != WIRE_VERSION:
            raise ValueError("Response protocol mismatch")
        return result

    def _identity(self):
        return {"protocol_version": WIRE_VERSION, "session_id": self.session_id, "episode_id": self.episode_id}

    def reset(self, episode_id, seed=None):
        if type(episode_id) is not int or episode_id <= self.episode_id:
            raise ValueError("Episode ids must increase")
        payload = {**self._identity(), "episode_id": episode_id, "seed": seed}
        result = self._request("/reset", payload)
        self._check_echo(result, {k: v for k, v in payload.items() if k != "seed"})
        self.episode_id, self.last_control_step = episode_id, -1

    @staticmethod
    def _check_echo(response, expected):
        if any(response.get(k) != v for k, v in expected.items()):
            raise ValueError("Stale or mismatched DP response")

    def infer(self, rgb, state, control_step):
        if self.episode_id < 0 or type(control_step) is not int or control_step <= self.last_control_step:
            raise ValueError("Reset first; inference control steps must increase")
        identity = {**self._identity(), "request_id": self.request_id, "control_step": control_step}
        result = self._request(
            "/infer", {**identity, "rgb": encode_rgb(rgb), "state": array(state, (88,), "state").tolist()}
        )
        self._check_echo(result, identity)
        action = array(result["action_chunk"], (40, 125), "action_chunk")
        self.request_id += 1
        self.last_control_step = control_step
        return action, {k: result[k] for k in (*identity, "inference_time_ms")}

    def close(self):
        if self.episode_id >= 0:
            identity = self._identity()
            self._check_echo(self._request("/close", identity), identity)
            self.episode_id = -1
