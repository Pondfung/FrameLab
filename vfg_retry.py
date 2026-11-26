"""Conservative acceptance checks for one VFG retry at the same time."""

from functools import cache

import numpy as np

from vfg_review import MIN_MOTION, distance


VERSION = 1
TRIGGER_COPY_DELTA = 0.25


def copied_anchor(left, right, candidate, time):
    if not 0.2 <= float(time) <= 0.8:
        return None
    motion = distance(left, right)
    if motion < MIN_MOTION:
        return None
    nearest = min(distance(left, candidate), distance(right, candidate))
    if nearest <= TRIGGER_COPY_DELTA:
        return {"anchor_delta": round(motion, 3), "forward_delta": round(nearest, 3)}
    return None


@cache
def motion_backend():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("VFG retry requires OpenCV. Run python setup_runtime.py deps.") from exc
    cv2.setNumThreads(2)
    return cv2


def track(cv2, left, right, points):
    options = dict(winSize=(19, 19), maxLevel=3)
    dest, first, _ = cv2.calcOpticalFlowPyrLK(left, right, points, None, **options)
    back, second, _ = cv2.calcOpticalFlowPyrLK(right, left, dest, None, **options)
    reliable = first.ravel().astype(bool) & second.ravel().astype(bool)
    reliable &= np.linalg.norm(back[:, 0] - points[:, 0], axis=1) <= 1
    return dest, reliable


def verify_retry(left, right, candidate, time):
    nearest = min(distance(left, candidate), distance(right, candidate))
    stats = {"retry_delta": round(nearest, 3), "anchor_tracks": 0, "retry_tracks": 0, "in_time_tracks": 0}
    if nearest < 1.5:
        return "still_near_anchor", stats

    cv2 = motion_backend()

    left, right, candidate = (np.rint(image).astype(np.uint8) for image in (left, right, candidate))
    points = cv2.goodFeaturesToTrack(left, 240, 0.01, 5)
    if points is None:
        return "unverified_motion", stats
    end, reliable = track(cv2, left, right, points)
    vector = end[:, 0] - points[:, 0]
    reliable &= np.linalg.norm(vector, axis=1) >= 3
    stats["anchor_tracks"] = int(reliable.sum())
    if not reliable.any():
        return "unverified_motion", stats
    points, end, vector = points[reliable], end[reliable], vector[reliable]
    mid, from_left = track(cv2, left, candidate, points)
    mid_from_right, from_right = track(cv2, right, candidate, end)
    consistent = from_left & from_right & (np.linalg.norm(mid[:, 0] - mid_from_right[:, 0], axis=1) <= 2)
    delta = mid[:, 0] - points[:, 0]
    length = np.linalg.norm(vector, axis=1)
    along = np.sum(delta * vector, axis=1) / length**2
    across = np.linalg.norm(delta - along[:, None] * vector, axis=1)
    on_time = consistent & (np.abs(along - float(time)) <= 0.20) & (across <= np.maximum(1.5, length * 0.20))
    stats["retry_tracks"] = int(consistent.sum())
    stats["in_time_tracks"] = votes = int(on_time.sum())
    # Require evidence from more than one small region of the image.
    cells = np.floor_divide(points[on_time, 0], [80, 45]).astype(int)
    stats["support_cells"] = len(np.unique(cells, axis=0))
    if votes >= 18 and 4 * votes >= 3 * stats["retry_tracks"] and 2 * votes >= stats["anchor_tracks"]:
        if stats["support_cells"] >= 3:
            return "accepted", stats
    return "unverified_motion", stats
