"""
Feature extraction layer for the AI Coach Cricketer pipeline.

Turns raw MediaPipe pose landmarks into meaningful, normalised
biomechanical features, then aggregates them across the whole video
(means / minima / maxima / standard deviations / ranges of motion).

Angles are computed in pixel space so the result is independent of the
video's aspect ratio. Every calculation is guarded against missing or
low-confidence landmarks, so a partial pose never crashes the pipeline.
"""

import math
import statistics


# ----------------------------------------------------------------------
# MediaPipe Pose landmark indices (33 landmarks, PoseLandmarker)
# ----------------------------------------------------------------------
NOSE = 0
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28
L_HEEL = 29
R_HEEL = 30
L_FOOT_INDEX = 31
R_FOOT_INDEX = 32

# Minimum visibility a landmark needs to be considered reliable.
MIN_VISIBILITY = 0.5


# ----------------------------------------------------------------------
# Low-level geometry helpers
# ----------------------------------------------------------------------
def is_visible(landmark):
    """A landmark is usable if MediaPipe marked it visible enough."""
    if landmark is None:
        return False
    # visibility may be None in some exports; treat missing as visible
    if landmark.visibility is None:
        return True
    return landmark.visibility >= MIN_VISIBILITY


def to_pixel(landmark, width, height):
    """Convert a normalized landmark to pixel coordinates."""
    return (landmark.x * width, landmark.y * height)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def calculate_angle(a, b, c):
    """
    Angle at point `b` formed by segments b-a and b-c, in degrees.
    a, b, c are (x, y) tuples.
    """
    try:
        angle = math.degrees(
            math.atan2(c[1] - b[1], c[0] - b[0])
            - math.atan2(a[1] - b[1], a[0] - b[0])
        )
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle
        return angle
    except (TypeError, ZeroDivisionError, ValueError):
        return None


def angle_from_vertical(a, b):
    """
    Angle (degrees) of segment a->b relative to the vertical axis.
    0 = perfectly upright, 90 = horizontal.
    """
    try:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        # angle from vertical
        rad = math.atan2(dx, -dy)
        return math.degrees(rad)
    except (TypeError, ValueError):
        return None


def angle_from_horizontal(a, b):
    """
    Angle (degrees) of segment a->b relative to the horizontal axis.
    0 = level, +90 = pointing straight up.
    """
    try:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return math.degrees(math.atan2(-dy, dx))
    except (TypeError, ValueError):
        return None


def midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


# ----------------------------------------------------------------------
# Per-frame feature extraction
# ----------------------------------------------------------------------
class FeatureExtractor:
    """
    Extracts one feature dictionary from a single frame's landmarks.

    The landmarks are expected to be the raw MediaPipe objects (with
    .x, .y, .visibility). The caller supplies the frame dimensions so
    normalized coordinates can be converted to pixel space for correct
    angle and distance math.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        # torso length acts as a scale-normalising reference
        self._torso = None

    def extract(self, landmarks):
        """
        Return a dict of features for this frame, or None if too few
        landmarks are visible to say anything meaningful.
        """
        if landmarks is None:
            return None

        p = {}
        for idx, lm in enumerate(landmarks):
            p[idx] = lm

        # ----- pixel-space points (only if visible) -----
        def pt(idx):
            if not is_visible(p.get(idx)):
                return None
            return to_pixel(p[idx], self.width, self.height)

        l_shoulder = pt(L_SHOULDER)
        r_shoulder = pt(R_SHOULDER)
        l_elbow = pt(L_ELBOW)
        r_elbow = pt(R_ELBOW)
        l_wrist = pt(L_WRIST)
        r_wrist = pt(R_WRIST)
        l_hip = pt(L_HIP)
        r_hip = pt(R_HIP)
        l_knee = pt(L_KNEE)
        r_knee = pt(R_KNEE)
        l_ankle = pt(L_ANKLE)
        r_ankle = pt(R_ANKLE)
        l_foot = pt(L_FOOT_INDEX)
        r_foot = pt(R_FOOT_INDEX)
        nose = pt(NOSE)

        # Need at least the torso to normalise distances
        if l_shoulder is None or r_shoulder is None or \
           l_hip is None or r_hip is None:
            return None

        mid_shoulder = midpoint(l_shoulder, r_shoulder)
        mid_hip = midpoint(l_hip, r_hip)
        torso = distance(mid_shoulder, mid_hip)
        if torso <= 0:
            torso = 1.0
        self._torso = torso

        # ----- joint angles (pixel space) -----
        feats = {
            "l_elbow": calculate_angle(l_shoulder, l_elbow, l_wrist) if (l_shoulder and l_elbow and l_wrist) else None,
            "r_elbow": calculate_angle(r_shoulder, r_elbow, r_wrist) if (r_shoulder and r_elbow and r_wrist) else None,
            "l_knee": calculate_angle(l_hip, l_knee, l_ankle) if (l_hip and l_knee and l_ankle) else None,
            "r_knee": calculate_angle(r_hip, r_knee, r_ankle) if (r_hip and r_knee and r_ankle) else None,
            "l_hip": calculate_angle(l_shoulder, l_hip, l_knee) if (l_shoulder and l_hip and l_knee) else None,
            "r_hip": calculate_angle(r_shoulder, r_hip, r_knee) if (r_shoulder and r_hip and r_knee) else None,
        }

        # ----- alignment & posture -----
        feats["shoulder_slope"] = angle_from_horizontal(l_shoulder, r_shoulder)
        feats["hip_slope"] = angle_from_horizontal(l_hip, r_hip)
        feats["torso_angle"] = angle_from_vertical(mid_shoulder, mid_hip)

        # ----- head position relative to body -----
        if nose is not None:
            feats["head_x_offset"] = (nose[0] - mid_hip[0]) / torso
            feats["head_height_rel"] = (mid_hip[1] - nose[1]) / torso
        else:
            feats["head_x_offset"] = None
            feats["head_height_rel"] = None

        # ----- stance & balance -----
        # stance width: distance between feet, normalised by torso length
        if l_foot is not None and r_foot is not None:
            feats["stance_width"] = distance(l_foot, r_foot) / torso
        else:
            feats["stance_width"] = None

        # balance offset: horizontal shift of the body centre (mid-hip)
        # relative to the centre of the support base (mid-foot)
        support = None
        if l_foot is not None and r_foot is not None:
            support = midpoint(l_foot, r_foot)
        elif l_foot is not None:
            support = l_foot
        elif r_foot is not None:
            support = r_foot
        if support is not None:
            feats["balance_offset"] = (mid_hip[0] - support[0]) / torso
        else:
            feats["balance_offset"] = None

        # ----- arm extension / reach (wrist distance from shoulder) -----
        feats["l_arm_reach"] = distance(l_shoulder, l_wrist) / torso if (l_shoulder and l_wrist) else None
        feats["r_arm_reach"] = distance(r_shoulder, r_wrist) / torso if (r_shoulder and r_wrist) else None

        # ----- hand (wrist) height relative to shoulder line -----
        if l_wrist is not None and r_wrist is not None:
            feats["wrist_height"] = ((mid_shoulder[1] - min(l_wrist[1], r_wrist[1])) / torso)
        else:
            feats["wrist_height"] = None

        # ----- how many of the key landmarks are visible -----
        feats["visible_count"] = sum(1 for lm in p.values() if is_visible(lm))

        return feats


# ----------------------------------------------------------------------
# Frame-by-frame aggregation
# ----------------------------------------------------------------------
class FeatureAggregator:
    """
    Accumulates per-frame features and produces summary statistics
    (mean / std / min / max / range) for every numeric feature.
    """

    def __init__(self):
        self._series = {}
        self.frame_count = 0

    def add(self, feats):
        if feats is None:
            return
        self.frame_count += 1
        for key, value in feats.items():
            if value is None:
                continue
            self._series.setdefault(key, []).append(value)

    def series(self):
        """Return the raw per-frame feature lists (for time-series analysis)."""
        return {k: list(v) for k, v in self._series.items()}

    def summarize(self):
        """
        Return dict of {feature: {mean, std, min, max, range}} for each
        feature that had at least one usable value.
        """
        summary = {}
        for key, values in self._series.items():
            if not values:
                continue
            mean = statistics.fmean(values)
            try:
                std = statistics.pstdev(values) if len(values) > 1 else 0.0
            except statistics.StatisticsError:
                std = 0.0
            summary[key] = {
                "mean": round(mean, 2),
                "std": round(std, 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "range": round(max(values) - min(values), 2),
                "n": len(values),
            }
        return summary
