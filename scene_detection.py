from collections import deque
from queue import SimpleQueue
import subprocess
from threading import Thread
from time import monotonic

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from video_lab import command


ALGORITHM = "ffmpeg-scene-local-motion-v1"
WIDTH, HEIGHT = 160, 90
PATCH = 12
CANDIDATE_SCORE = 0.18
GRID_X, GRID_Y = (grid.ravel() for grid in np.meshgrid(np.arange(28, 131, 17), np.arange(24, 61, 12)))
SHIFT_X, SHIFT_Y = (grid.ravel() for grid in np.meshgrid(np.arange(-18, 19, 2), np.arange(-18, 19, 2)))


def read_scores(pipe, scores, diagnostics):
    try:
        for line in pipe:
            if line.startswith(b"lavfi.scene_score="):
                scores.put(float(line.split(b"=", 1)[1]))
            elif not line.startswith(b"frame:"):
                diagnostics.append(line)
    finally:
        scores.put(None)


def thumbnails(source, stream):
    # Pair every decoded thumbnail with its FFmpeg score, including duplicates.
    filter_chain = (f"scale={WIDTH}:{HEIGHT}:flags=area,select='gte(scene,0)',"
                    r"metadata=mode=print:key=lavfi.scene_score:file='pipe\:2':direct=1,format=rgb24")
    args = command("ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-xerror", "-noautorotate",
                   "-threads", "4", "-i", source, "-map", f"0:{stream}", "-an", "-vf", filter_chain,
                   "-fps_mode", "passthrough", "-threads:v", "1", "-f", "rawvideo", "pipe:1")
    scores = SimpleQueue()
    diagnostics = deque(maxlen=12)
    frame_bytes = WIDTH * HEIGHT * 3
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        reader = Thread(target=read_scores, args=(process.stderr, scores, diagnostics))
        reader.start()
        try:
            while raw := process.stdout.read(frame_bytes):
                score = scores.get()
                if len(raw) != frame_bytes or score is None:
                    raise RuntimeError("Incomplete FFmpeg scene analysis: " + b"".join(diagnostics).decode(errors="replace"))
                rgb = np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
                yield rgb.mean(axis=-1, dtype=np.float32), score
            if process.wait() or scores.get() is not None:
                raise RuntimeError("FFmpeg scene analysis failed: " + b"".join(diagnostics).decode(errors="replace"))
        finally:
            if process.poll() is None:
                process.kill()
            reader.join()


def normalized_patches(patches):
    flat = patches.reshape(*patches.shape[:-2], PATCH * PATCH)
    centered = flat - flat.mean(axis=-1, keepdims=True)
    length = np.linalg.norm(centered, axis=-1)
    return centered / np.maximum(length[..., None], 1), length / PATCH


def translated_support(reference, search):
    anchors, texture = normalized_patches(reference[GRID_Y, GRID_X])
    candidates, candidate_texture = normalized_patches(search[GRID_Y[:, None] + SHIFT_Y, GRID_X[:, None] + SHIFT_X])
    similarity = np.einsum("ijk,ik->ij", candidates, anchors)
    similarity[candidate_texture < 3] = -1
    textured = texture >= 4
    return np.count_nonzero(textured & (similarity.max(axis=1) >= .78)) / max(1, np.count_nonzero(textured))


def aligned_support(left, right):
    # Use interior tiles so a stationary HUD or letterbox alone cannot veto a cut.
    a, contrast_a = normalized_patches(left[12:72:PATCH, 24:132:PATCH])
    b, contrast_b = normalized_patches(right[12:72:PATCH, 24:132:PATCH])
    return int(np.count_nonzero((contrast_a >= 4) & (contrast_b >= 4) & (np.sum(a * b, axis=-1) >= .84)))


def luminance_change(left, right):
    return float(np.abs(left - right).mean())


def classify(index, score, before, after):
    left, right = before[-1], after[0]
    change = luminance_change(left, right)
    nearby = max(1, luminance_change(before[-2] if len(before) > 1 else left, left),
                 luminance_change(right, after[min(1, len(after) - 1)]))
    result = {"frame": index, "scene_score": score, "luma_change": round(change, 3),
              "relative_change": round(change / nearby, 3)}
    isolated_change = change >= 1.6 * nearby
    if change < 8 or (score < .4 and change < 40 and not isolated_change):
        return dict(result, decision="continuous_change")
    bridge = min(luminance_change(a, b) for a in before for b in after) / change
    result["bridge"] = round(bridge, 3)
    if bridge <= .48:
        return dict(result, decision="brief_change")
    a, b = (sliding_window_view(frame, (PATCH, PATCH)) for frame in (left, right))
    motion = (translated_support(a, b) + translated_support(b, a)) / 2
    aligned = aligned_support(a, b)
    result.update(motion_support=round(motion, 3), aligned_tiles=aligned)
    if motion > .36 or aligned >= 3 or (aligned > 0 and motion >= .22):
        return dict(result, decision="matching_texture")
    # An abrupt transition embedded in motion needs stronger texture loss
    # than an isolated jump. Adjacent weaker jumps often belong to the same effect.
    if (score < .4 and not isolated_change and motion >= .22) or (change < .7 * nearby and motion >= .1):
        return dict(result, decision="continuous_change")
    return dict(result, decision="cut")


def scan_cuts(source, video):
    start = monotonic()
    previous = deque(maxlen=3)
    following = deque()
    candidates = []
    total = 0

    def check_pending():
        index, frame, score = following[0]
        if previous and score >= CANDIDATE_SCORE:
            candidates.append(classify(index, score, list(previous), [sample for _, sample, _ in following]))
        previous.append(frame)
        following.popleft()

    for index, (frame, score) in enumerate(thumbnails(source, video["index"])):
        total += 1
        following.append((index, frame, score))
        if len(following) == 3:
            check_pending()
    while following:
        check_pending()
    if total != video["decoded_frames"]:
        raise RuntimeError(f"Scene analysis read {total} frames, expected {video['decoded_frames']}")
    return dict(algorithm=ALGORITHM, scanned_frames=total, seconds=round(monotonic() - start, 3),
                detected=[row["frame"] for row in candidates if row["decision"] == "cut"], candidates=candidates)
