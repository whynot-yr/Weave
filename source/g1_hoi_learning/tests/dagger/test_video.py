"""Video lifetime, terminal-frame alignment, and real CPU MP4 encoding."""

import numpy as np
import pytest
from g1_hoi_learning.algorithms.dagger.video import ClipVideoRecorder, VideoConfig


class FakeWriter:
    def __init__(self, path, fps):
        self.path, self.fps = path, fps
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.closed = True
        self.path.write_bytes(b"encoded")


def test_video_terminal_frame_is_saved_before_reset(tmp_path):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    recorder = ClipVideoRecorder(tmp_path, VideoConfig(width=16, height=16), 0.02, lambda: frame, FakeWriter)
    recorder.start(7, "../same:motion")
    writer = recorder.writer
    for step in range(1, 6):
        frame.fill(step * 10)
        recorder.record_post_step(terminal=step == 5)
    # The environment now automatically resets. No reset frame may enter this clip.
    frame.fill(250)
    recorder.record_post_step(terminal=False)
    result = recorder.finish(complete=True)
    assert [int(rgb[0, 0, 0]) for rgb in writer.frames] == [0, 20, 40, 50]
    assert result["video_frame_steps"] == [0, 2, 4, 5]
    assert result["video_complete"] and result["video_fps"] == 25
    assert result["video_frames"] == 4
    assert result["video_simulation_seconds"] == pytest.approx(0.1)
    assert (tmp_path / result["video_path"]).is_file() and writer.closed
    assert not list(tmp_path.glob("videos/*.partial.mp4"))
    assert recorder.finish(complete=False) is None
    recorder.start(8, "../same:motion")
    other = recorder.finish(complete=False)
    assert other["video_path"] != result["video_path"]
    assert not other["video_complete"]


def test_terminal_on_stride_boundary_is_not_duplicated(tmp_path):
    frame = np.ones((16, 16, 3), dtype=np.uint8)
    recorder = ClipVideoRecorder(tmp_path, VideoConfig(width=16, height=16), 0.02, lambda: frame, FakeWriter)
    recorder.start(0, "motion")
    recorder.record_post_step(terminal=False)
    recorder.record_post_step(terminal=True)
    assert recorder.finish(complete=True)["video_frame_steps"] == [0, 2]


def test_repeated_clip_trials_have_distinct_videos(tmp_path):
    frame = np.ones((16, 16, 3), dtype=np.uint8)
    recorder = ClipVideoRecorder(tmp_path, VideoConfig(width=16, height=16), 0.02, lambda: frame, FakeWriter)
    paths = []
    for trial_id in (0, 1):
        recorder.start(0, "same_clip", trial_id=trial_id)
        paths.append(recorder.finish(complete=False)["video_path"])
    assert paths[0] != paths[1]
    assert all((tmp_path / path).is_file() for path in paths)


def test_interrupted_video_is_closed_and_marked_incomplete(tmp_path):
    frame = np.ones((16, 16, 3), dtype=np.uint8)
    recorder = ClipVideoRecorder(tmp_path, VideoConfig(width=16, height=16), 0.02, lambda: frame, FakeWriter)
    recorder.start(0, "motion")
    with pytest.raises(RuntimeError, match="previous video"):
        recorder.start(1, "motion")
    for _ in range(3):
        recorder.record_post_step(terminal=False)
    writer = recorder.writer
    result = recorder.finish(complete=False)
    assert writer.closed and not recorder.active
    assert result["video_frame_steps"] == [0, 2] and not result["video_complete"]


@pytest.mark.parametrize("settings", [{"width": 0}, {"height": 17}, {"frame_stride": 0}, {"enabled": "false"}])
def test_invalid_video_configuration(settings):
    with pytest.raises(ValueError):
        VideoConfig(**settings)


def test_invalid_rgb_does_not_open_encoder(tmp_path):
    recorder = ClipVideoRecorder(
        tmp_path,
        VideoConfig(width=16, height=16),
        0.02,
        lambda: np.zeros((16, 16, 4), dtype=np.uint8),
        FakeWriter,
    )
    with pytest.raises(ValueError, match="RGB frame"):
        recorder.start(0, "motion")
    assert not recorder.active


def test_real_mp4_can_be_decoded_with_terminal_frame(tmp_path):
    import imageio.v2 as imageio

    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    recorder = ClipVideoRecorder(tmp_path, VideoConfig(width=64, height=48), 0.02, lambda: frame)
    frame[:, :, 0] = 200
    recorder.start(0, "real")
    for step in range(1, 6):
        frame[:, :, 0] = 0
        frame[:, :, 1] = step * 40
        recorder.record_post_step(terminal=step == 5)
    result = recorder.finish(complete=True)
    reader = imageio.get_reader(tmp_path / result["video_path"], format="FFMPEG")
    try:
        assert reader.count_frames() == result["video_frames"] == 4
        assert reader.get_meta_data()["fps"] == 25
        first, last = reader.get_data(0), reader.get_data(3)
        assert first.shape == last.shape == (48, 64, 3)
        assert first[:, :, 0].mean() > 180 and last[:, :, 1].mean() > 180
        assert last[:, :, 0].mean() < 10
    finally:
        reader.close()
