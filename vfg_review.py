"""Inspect VFG output before scaling or encoding."""

from collections import Counter
from fractions import Fraction

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


VERSION = 2
SUPPORTED_VERSIONS = (1, VERSION)
MIN_MOTION = 2.0
MAX_COPY = 0.85
COPY_FRACTION = 0.10


def sample_grid(rgb, width=320, height=180):
    # Sample at the same pixel locations in source and generated RGB, using BT.709 weights.
    rows = (np.arange(height) * 2 + 1) * rgb.shape[0] // (2 * height)
    cols = (np.arange(width) * 2 + 1) * rgb.shape[1] // (2 * width)
    return rows, cols


def brightness(pixels, scale):
    return np.einsum("ijk,k->ij", pixels, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)) * scale


def distance(first, second):
    return float(np.abs(first - second).mean())


def match_patch(image, patch, y, x, radius):
    half = patch.shape[0] // 2
    area = image[y-radius-half:y+radius+half+1, x-radius-half:x+radius+half+1]
    windows = sliding_window_view(area, patch.shape)
    errors = np.abs(windows - patch).mean(axis=(-1, -2))
    row, col = np.unravel_index(errors.argmin(), errors.shape)
    best = float(errors[row, col])
    if best > 8 or min(row, col, 2 * radius - row, 2 * radius - col) == 0:
        return None
    # Competing distant matches indicate repeated texture or an ambiguous location.
    errors[max(0, row-1):row+2, max(0, col-1):col+2] = np.inf
    if errors.min() < best + 1.5:
        return None
    return np.array([row - radius, col - radius])


def backward_motion(left, right, images, comparisons):
    # Use sparse textured patches. Unmatched/occluded patches contribute no vote.
    images = [image[::2, ::2] for image in images]
    left, right = left[::2, ::2], right[::2, ::2]
    radius, half = 10, 4
    votes = Counter()
    tracks = Counter()
    for y in (22, 45, 67):
        for x in (25, 61, 97, 133):
            patch = left[y-half:y+half+1, x-half:x+half+1]
            if patch.std() < 10:
                continue
            motion = match_patch(right, patch, y, x, radius)
            if motion is None or (length := float(np.linalg.norm(motion))) < 3:
                continue
            matched = [match_patch(image, patch, y, x, radius) for image in images]
            for i in comparisons:
                a, b = matched[i-1:i+1]
                if a is None or b is None:
                    continue
                delta = b - a
                if abs(delta[0] * motion[1] - delta[1] * motion[0]) > length * 2:
                    continue
                tracks[i] += 1
                if delta @ motion / length < -1.5:
                    votes[i] += 1
    return {i: (votes[i], tracks[i]) for i in comparisons if votes[i] >= 3 and votes[i] * 3 >= tracks[i] * 2}


class BlockReview:
    def __init__(self, right, stop, first, first_out, output_rate, source_offset, *,
                 version=VERSION, retry_version=None):
        self.right = right
        self.stop = stop
        self.first = first
        self.tick = first_out
        self.output_rate = output_rate
        self.source_offset = source_offset
        self.roles = Counter()
        self.events = {}
        self.version = version
        self.retry_version = retry_version
        self.retries = []
        self.index = None
        self.anchor_tick = None
        self.group = []
        self.grid = None

    def __call__(self, frames, position, sample, *, retry=None):
        index = int(position)
        tick = self.tick
        self.tick += 1
        if index != self.index:
            self.flush()
            self.index = index
            self.anchor_tick = None
        if position == index:
            self.roles["source"] += 1
            self.anchor_tick = tick
        elif sample is None:
            self.roles["cut_hold" if self.right < self.stop else "tail_hold"] += 1
        else:
            self.roles["generated"] += 1
            if self.grid is None:
                self.grid = sample_grid(frames[index])
            rows, cols = self.grid
            if not self.group:
                self.left, self.right_image = (
                    brightness(frame[rows[:, None], cols], 1 / 257) for frame in frames[index:index+2]
                )
            time = position - index
            image = sample(rows, cols)
            accepted = False
            if retry is not None:
                from vfg_retry import copied_anchor, verify_retry

                evidence = copied_anchor(self.left, self.right_image, image, time)
                if evidence is not None:
                    candidate = retry(rows, cols)
                    status, check = verify_retry(self.left, self.right_image, candidate, time)
                    self.retries.append(dict(self.frame_details(tick, time), status=status, **evidence, **check))
                    if status == "accepted":
                        image = candidate
                        accepted = True
            self.group.append((tick, time, image))
            return accepted
        return False

    def frame_details(self, tick, offset):
        anchor = self.source_offset + self.first + self.index
        return dict(output_frame=tick, time=str(tick / self.output_rate), role="generated",
                    source_frames=[anchor, anchor + 1], timestep=str(offset))

    def add(self, item, reason, **evidence):
        tick, offset, _ = item
        event = self.events.setdefault(tick, dict(self.frame_details(tick, offset), reasons={}))
        event["reasons"][reason] = evidence

    def record_anchor_echo(self, i):
        # Use only consecutive VFG output slots in the same source interval.
        # An original frame can be evidence, never the frame being reported.
        if i == 0:
            return
        item = self.group[i]
        tick, _, image = item
        middle_tick, _, middle = self.group[i-1]
        if i == 1:
            if self.anchor_tick != tick - 2:
                return
            earlier = self.left
        else:
            earlier = self.group[i-2][2]
        near = distance(earlier, image)
        forward = distance(earlier, middle)
        backward = distance(middle, image)
        if (min(forward, backward) > 1 and near <= min(1, 0.5 * min(forward, backward))
                and abs(forward - backward) <= 0.6 * max(forward, backward)):
            self.add(item, "anchor_echo", echoed_output_frame=tick - 2, intervening_output_frame=middle_tick,
                     echo_delta=round(near, 3), forward_delta=round(forward, 3), backward_delta=round(backward, 3))

    def flush(self):
        if not self.group:
            return
        motion = distance(self.left, self.right_image)
        if motion < MIN_MOTION:
            role = "static_source" if motion <= 0.25 else "low_motion"
            # Count these among the generated slots above and separately for coverage.
            self.roles[role] += len(self.group)
        else:
            limit = min(MAX_COPY, COPY_FRACTION * motion)
            copied = {}
            for i, item in enumerate(self.group):
                image = item[2]
                distances = [distance(image, anchor) for anchor in (self.left, self.right_image)]
                closest = int(distances[1] < distances[0])
                if distances[closest] <= limit:
                    copied[i] = closest
                    self.add(item, "anchor_copy", anchor=self.source_offset + self.first + self.index + closest,
                             anchor_delta=round(motion, 3), candidate_delta=round(distances[closest], 3))
                    if self.version >= 2:
                        self.record_anchor_echo(i)
            comparisons = []
            for i, (a, b) in enumerate(zip(self.group, self.group[1:]), start=1):
                delta = distance(a[2], b[2])
                if delta <= limit:
                    for item in (a, b):
                        self.add(item, "repeated_generated", pair=[a[0], b[0]], candidate_delta=round(delta, 3))
                elif copied.get(i-1) == 1 and copied.get(i) == 0:
                    self.add(b, "anchor_backtrack", previous_output_frame=a[0])
                else:
                    comparisons.append(i)
            if comparisons:
                backward = backward_motion(self.left, self.right_image, [item[2] for item in self.group], comparisons)
                for i, (votes, tracks) in backward.items():
                    self.add(self.group[i], "motion_backtrack", previous_output_frame=self.group[i-1][0],
                             backward_tracks=votes, tracked_patches=tracks)
        self.group.clear()

    def summary(self):
        self.flush()
        details = {"roles": dict(self.roles), "suspects": list(self.events.values())}
        if self.retry_version is not None:
            details.update(retry_version=self.retry_version, retries=self.retries)
        return details


def report(blocks, *, stage, input_rate, output_rate, source_start, cuts, sdk_shot_detection,
           version=VERSION, retry_version=None):
    roles = Counter()
    events = {}
    retries = []
    for block in blocks:
        roles.update(block["roles"])
        if retry_version is not None:
            retries.extend(block["retries"])
        for event in block["suspects"]:
            events[event["output_frame"]] = event
    suspects = [events[tick] for tick in sorted(events)]
    runs = []
    copies = {}
    output_rate = Fraction(output_rate)
    for event in suspects:
        reasons = event["reasons"]
        tick = event["output_frame"]
        if "anchor_copy" in reasons:
            anchor = reasons["anchor_copy"]["anchor"]
            copies.setdefault(anchor, set()).add(tick)
            anchor_tick = (anchor - source_start) * output_rate / input_rate
            if anchor_tick.denominator == 1 and anchor_tick >= 0:
                copies[anchor].add(int(anchor_tick))
    for anchor, ticks in copies.items():
        first = last = min(ticks)
        for tick in [*sorted(ticks - {first}), max(ticks) + 2]:
            if tick != last + 1:
                if last - first >= 2:
                    runs.append(dict(anchor=anchor, first_output_frame=first, last_output_frame=last))
                first = tick
            last = tick
    for reason in ("cut_hold", "tail_hold", "source", "generated", "static_source", "low_motion"):
        roles.setdefault(reason, 0)
    details = dict(version=version, stage=stage, source_start=source_start,
                   input_rate=str(input_rate), output_rate=str(output_rate),
                   effective_cuts=cuts, sdk_shot_detection=sdk_shot_detection, roles=dict(roles),
                   reason_counts=dict(Counter(key for event in suspects for key in event["reasons"])),
                   anchor_holds=sorted(runs, key=lambda run: run["first_output_frame"]), suspects=suspects)
    if retry_version is not None:
        details.update(retry_version=retry_version, retry_counts=dict(Counter(item["status"] for item in retries)),
                       retries=sorted(retries, key=lambda item: item["output_frame"]))
    return details
