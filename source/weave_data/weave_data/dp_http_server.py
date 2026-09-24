"""Arena-style HTTP inference server, adapted to WEAVE and current LeRobot APIs."""

import threading
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from .dp_protocol import (
    MAX_BODY_BYTES,
    WIRE_VERSION,
    array,
    decode_rgb,
    json_bytes,
    read_json,
    validate_health,
    validate_identity,
)


class BusyError(RuntimeError):
    pass


class DPService:
    def __init__(self, predictor, model_id):
        self.predictor = predictor
        # Do not send potentially large per-episode metadata on every health check.
        protocol = {k: v for k, v in predictor.protocol.items() if k not in {"episodes", "source"}}
        self.health = {
            "protocol_version": WIRE_VERSION,
            "status": "ready",
            "model_id": model_id,
            "state_shape": [88],
            "action_shape": [40, 125],
            "image_shape": [224, 224, 3],
            "n_obs_steps": 1,
            "data_protocol": protocol,
        }
        validate_health(self.health)
        self.session = None
        self.episode = self.last_request = self.last_step = -1
        self.lock = threading.Lock()

    def dispatch(self, path, payload):
        validate_identity(payload, inference=path == "/infer")
        if not self.lock.acquire(blocking=False):
            raise BusyError("Inference/reset already in progress")
        try:
            session, episode = payload["session_id"], payload["episode_id"]
            if self.session is not None and self.session != session:
                raise BusyError("Another session owns this service; close it or restart the service")
            identity = {"protocol_version": WIRE_VERSION, "session_id": session, "episode_id": episode}
            if path == "/reset":
                if self.session is not None and episode <= self.episode:
                    raise ValueError("Episode id must increase")
                seed = payload.get("seed")
                if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
                    raise ValueError("Invalid seed")
                self.predictor.reset(seed)
                self.session, self.episode = session, episode
                self.last_request = self.last_step = -1
                return identity
            if self.session != session or self.episode != episode:
                raise ValueError("Reset first or stale episode")
            if path == "/close":
                self.predictor.reset()
                self.session = None
                self.episode = self.last_request = self.last_step = -1
                return identity
            if path != "/infer":
                raise ValueError("Unknown endpoint")
            if payload["request_id"] <= self.last_request or payload["control_step"] <= self.last_step:
                raise ValueError("Duplicate/stale request")
            rgb = torch.from_numpy(decode_rgb(payload["rgb"]))[None]
            state = torch.from_numpy(array(payload["state"], (88,), "state"))[None]
            start = time.perf_counter()
            action = self.predictor.predict_encoded(rgb, state).detach().cpu().numpy()
            action = array(action, (1, 40, 125), "prediction")[0]
            self.last_request, self.last_step = payload["request_id"], payload["control_step"]
            return {
                **identity,
                "request_id": self.last_request,
                "control_step": self.last_step,
                "action_chunk": action.tolist(),
                "inference_time_ms": (time.perf_counter() - start) * 1000,
            }
        finally:
            self.lock.release()


def make_server(service, host="127.0.0.1", port=8765):
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("V1 service must bind to loopback")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def reply(self, code, payload):
            body = json_bytes(payload)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            # A timed-out client has stopped control; a late response must not
            # turn into an unhandled worker exception or trigger another infer.
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def do_GET(self):
            self.reply(
                200 if self.path == "/health" else 404,
                service.health if self.path == "/health" else {"protocol_version": WIRE_VERSION, "error": "Not found"},
            )

        def do_POST(self):
            try:
                if self.path not in {"/infer", "/reset", "/close"}:
                    self.reply(404, {"protocol_version": WIRE_VERSION, "error": "Not found"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ValueError("Invalid/oversized request body")
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Chunked encoding not supported")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Truncated request")
                result = service.dispatch(self.path, read_json(raw))
            except BusyError as exc:
                self.reply(409, {"protocol_version": WIRE_VERSION, "error": str(exc)})
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                self.reply(400, {"protocol_version": WIRE_VERSION, "error": str(exc)})
            except Exception as exc:
                self.log_error("Inference error: %s", exc)
                self.reply(500, {"protocol_version": WIRE_VERSION, "error": "Inference failed; see service log"})
            else:
                self.reply(200, result)

    return ThreadingHTTPServer((host, port), Handler)
