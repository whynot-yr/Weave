import copy
import threading

import numpy as np
import pytest
import torch
from weave_data.dp_codec import encode_reference, ppo_reference_groups
from weave_data.dp_http_client import DPHttpClient
from weave_data.dp_http_server import BusyError, DPService, make_server
from weave_data.dp_protocol import WIRE_VERSION, decode_rgb, encode_rgb, read_json, validate_health
from weave_data.dp_reference_buffer import DPReferenceBuffer


class FakePredictor:
    def __init__(self):
        self.protocol = {
            "protocol": "weave-dp-local-reference-v1",
            "complete": True,
            "position_mode": "direct",
            "quaternion_order": "wxyz",
            "pose_frame": "current_actual_pelvis_at_window_start",
            "rotation_6d": "first_two_matrix_columns_row_major",
            "joint_names": [f"j{i}" for i in range(53)],
            "action_names": [f"j{i}" for i in range(41)],
            "body_names": [f"b{i}" for i in range(54)],
            "active_joint_indices": list(range(41)),
            "fps": 50.0,
        }
        self.resets = []

    def reset(self, seed=None):
        self.resets.append(seed)

    def predict_encoded(self, rgb, state):
        assert rgb.shape == (1, 224, 224, 3) and rgb.dtype == torch.uint8
        return torch.full((1, 40, 125), float(state[0, 0]))


@pytest.fixture
def service():
    state = DPService(FakePredictor(), "fake")
    server = make_server(state, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield state, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_http_session_and_replans(service):
    state, url = service
    client = DPHttpClient(url)
    rgb = np.full((224, 224, 3), 127, dtype=np.uint8)
    with pytest.raises(ValueError, match="Reset"):
        client.infer(rgb, np.ones(88), 0)
    client.reset(0, seed=42)
    for step in (0, 20, 40):
        action, details = client.infer(rgb, np.ones(88), step)
        assert action.shape == (40, 125) and (action == 1).all()
        assert details["control_step"] == step and details["episode_id"] == 0
    with pytest.raises(ValueError, match="increase"):
        client.infer(rgb, np.ones(88), 40)
    second = DPHttpClient(url)
    with pytest.raises(RuntimeError):
        second.reset(0)
    client.reset(1)
    client.infer(rgb, np.zeros(88), 0)
    assert state.predictor.resets[:2] == [42, None]
    client.close()
    second.reset(0)
    second.close()


def test_wire_validation_and_echo():
    rgb = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    np.testing.assert_array_equal(decode_rgb(encode_rgb(rgb)), rgb)
    broken = encode_rgb(rgb)
    broken["data_b64"] = "AAAA"
    with pytest.raises(ValueError):
        decode_rgb(broken)
    with pytest.raises(ValueError):
        read_json(b'{"number": NaN}')
    with pytest.raises(ValueError):
        DPHttpClient._check_echo({"episode_id": 1}, {"episode_id": 2})
    with pytest.raises(ValueError, match="loopback"):
        DPHttpClient("http://example.com")


def test_service_stale_and_invalid_inputs():
    service = DPService(FakePredictor(), "fake")
    identity = {"protocol_version": WIRE_VERSION, "session_id": "test", "episode_id": 0}
    service.dispatch("/reset", identity)
    payload = {
        **identity,
        "request_id": 0,
        "control_step": 0,
        "rgb": encode_rgb(np.zeros((224, 224, 3), dtype=np.uint8)),
        "state": [0.0] * 88,
    }
    invalid = copy.deepcopy(payload)
    invalid["state"][0] = float("nan")
    with pytest.raises(ValueError):
        service.dispatch("/infer", invalid)
    service.dispatch("/infer", payload)
    with pytest.raises(ValueError, match="stale"):
        service.dispatch("/infer", payload)
    service.dispatch("/reset", {**identity, "episode_id": 1})
    with pytest.raises(ValueError, match="stale"):
        service.dispatch("/infer", payload)


def test_reference_buffer_queries_and_reanchoring():
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    origin = torch.zeros(3)
    ref = {
        "joint_pos": torch.arange(40).float()[:, None].expand(-1, 53),
        "pelvis_pos": torch.randn(40, 3),
        "pelvis_quat": q.expand(40, -1),
        "object_pos": torch.randn(40, 3),
        "object_quat": q.expand(40, -1),
        "contact": torch.randint(-1, 2, (40, 54)).float(),
    }
    chunk = encode_reference(ref, origin, q)
    buffer = DPReferenceBuffer()
    assert buffer.needs_prediction(0)
    buffer.install(chunk, origin, q, 0)
    for j in range(20):
        position = torch.tensor([j * 0.01, 0.0, 0.0])
        rotation = torch.tensor([np.cos(j * 0.01), 0.0, 0.0, np.sin(j * 0.01)]).float()
        actual = buffer.reference_groups(position, rotation, j)
        expected = ppo_reference_groups(
            encode_reference(ref, position, rotation), offsets=tuple(j + x for x in (0, 5, 10, 15, 20))
        )
        for key in expected:
            torch.testing.assert_close(actual[key][0], expected[key], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(buffer.chunk, chunk)
    assert buffer.needs_prediction(20)
    with pytest.raises(RuntimeError):
        buffer.reference_groups(origin, q, 20)
    buffer.reset()
    assert buffer.needs_prediction(0)
    with pytest.raises(ValueError):
        DPReferenceBuffer(21)
    with pytest.raises(ValueError):
        buffer.install(torch.full((40, 125), float("nan")), origin, q, 0)


def test_http_timeout_and_busy(service, monkeypatch):
    state, url = service
    client = DPHttpClient(url)
    client.reset(0)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = state.predictor.predict_encoded

    def delayed(rgb, observation):
        entered.set()
        try:
            assert release.wait(timeout=5)
            return original(rgb, observation)
        finally:
            finished.set()

    monkeypatch.setattr(state.predictor, "predict_encoded", delayed)
    client.timeout = 0.1
    try:
        with pytest.raises(RuntimeError, match="request failed"):
            client.infer(np.zeros((224, 224, 3), dtype=np.uint8), np.zeros(88), 0)
        assert entered.is_set()
        assert client.request_id == 0 and client.last_control_step == -1
        with pytest.raises(BusyError):
            state.dispatch("/reset", {**client._identity(), "episode_id": 1})
    finally:
        release.set()
        assert finished.wait(timeout=5)
    # Serialize with the finishing worker, then close the session normally.
    with state.lock:
        pass
    client.timeout = 5
    client.close()


def test_health_and_response_validation(service, monkeypatch):
    state, url = service
    invalid = copy.deepcopy(state.health)
    invalid["data_protocol"]["position_mode"] = "delta"
    with pytest.raises(ValueError, match="Pose"):
        validate_health(invalid)
    client = DPHttpClient(url)
    client.reset(0)
    original = client._request

    def corrupted(path, payload=None):
        response = original(path, payload)
        if path == "/infer":
            response["action_chunk"] = [[0.0] * 125] * 39
        return response

    monkeypatch.setattr(client, "_request", corrupted)
    with pytest.raises(ValueError, match="action_chunk"):
        client.infer(np.zeros((224, 224, 3), dtype=np.uint8), np.zeros(88), 0)
    assert client.last_control_step == -1
    client.close()
