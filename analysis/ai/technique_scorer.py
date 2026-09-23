"""
Technique scoring (0-100).

Every score is a deterministic function of the aggregated biomechanical
measurements — there is no randomness. Each component maps a measured
feature toward an "ideal" value with a linear falloff to zero, then
clamps to 0-100. Missing measurements contribute a neutral 50 so a
partial pose never crashes or zeroes the score.

SCORING FORMULAS (documented):
  base_score(value, ideal, tolerance) = 100 * (1 - |value-ideal|/tolerance)
  clamped to [0, 100]; unknown values = 50.

  Posture  = 0.50*torso_upright + 0.30*head_centered + 0.20*head_height
  Balance  = 0.40*body_center + 0.30*stability + 0.30*stance_width
  Lower    = 0.50*knee_range_of_motion + 0.50*hip_range_of_motion
  Upper    = 0.60*elbow_range_of_motion + 0.40*arm_extension
  FollowTh = 0.60*arm_finish_extension + 0.40*high_finish

  Overall  = weighted mean of the five components.
"""


def base_score(value, ideal, tolerance):
    """Linear falloff score. Missing values score a neutral 50."""
    if value is None:
        return 50.0
    try:
        dist = abs(value - ideal)
        score = 100.0 * (1.0 - dist / tolerance)
        return max(0.0, min(100.0, score))
    except (TypeError, ZeroDivisionError, ValueError):
        return 50.0


def _m(summary, key, stat="mean"):
    """Read a stat from the aggregator summary, tolerantly."""
    try:
        value = summary[key][stat]
        return float(value)
    except (KeyError, TypeError, ValueError):
        return None


class TechniqueScorer:

    def score(self, summary):
        """summary: dict from FeatureAggregator.summarize()."""

        # ----- Posture -----
        torso_upright = base_score(_m(summary, "torso_angle"), ideal=5.0, tolerance=25.0)
        head_centered = base_score(_m(summary, "head_x_offset"), ideal=0.0, tolerance=0.5)
        head_height = base_score(_m(summary, "head_height_rel"), ideal=1.2, tolerance=1.0)
        posture = 0.50 * torso_upright + 0.30 * head_centered + 0.20 * head_height

        # ----- Balance -----
        body_center = base_score(_m(summary, "balance_offset"), ideal=0.0, tolerance=0.4)
        stability = base_score(_m(summary, "balance_offset", "std"), ideal=0.0, tolerance=0.3)
        stance = base_score(_m(summary, "stance_width"), ideal=1.0, tolerance=0.8)
        balance = 0.40 * body_center + 0.30 * stability + 0.30 * stance

        # ----- Lower body -----
        l_knee_rom = _m(summary, "l_knee", "range")
        r_knee_rom = _m(summary, "r_knee", "range")
        knee_rom = _mean(l_knee_rom, r_knee_rom)
        l_hip_rom = _m(summary, "l_hip", "range")
        r_hip_rom = _m(summary, "r_hip", "range")
        hip_rom = _mean(l_hip_rom, r_hip_rom)
        lower = 0.50 * base_score(knee_rom, ideal=40.0, tolerance=60.0) \
            + 0.50 * base_score(hip_rom, ideal=45.0, tolerance=70.0)

        # ----- Upper body / arms -----
        l_elbow_rom = _m(summary, "l_elbow", "range")
        r_elbow_rom = _m(summary, "r_elbow", "range")
        elbow_rom = _mean(l_elbow_rom, r_elbow_rom)
        l_reach = _m(summary, "l_arm_reach", "max")
        r_reach = _m(summary, "r_arm_reach", "max")
        arm_reach = _mean(l_reach, r_reach)
        upper = 0.60 * base_score(elbow_rom, ideal=100.0, tolerance=120.0) \
            + 0.40 * base_score(arm_reach, ideal=1.2, tolerance=1.0)

        # ----- Follow through -----
        l_elbow_max = _m(summary, "l_elbow", "max")
        r_elbow_max = _m(summary, "r_elbow", "max")
        elbow_finish = _mean(l_elbow_max, r_elbow_max)
        hand_high = _m(summary, "wrist_height", "max")
        follow = 0.60 * base_score(elbow_finish, ideal=165.0, tolerance=60.0) \
            + 0.40 * base_score(hand_high, ideal=1.2, tolerance=1.4)

        # ----- Overall (weighted mean) -----
        overall = (
            0.20 * posture
            + 0.20 * balance
            + 0.20 * lower
            + 0.20 * upper
            + 0.20 * follow
        )

        return {
            "overall": int(round(overall)),
            "posture": int(round(posture)),
            "balance": int(round(balance)),
            "lower_body": int(round(lower)),
            "upper_body": int(round(upper)),
            "follow_through": int(round(follow)),
            # keep the raw inputs for transparency / debugging
            "_raw": {
                "torso_upright": round(torso_upright, 1),
                "head_centered": round(head_centered, 1),
                "body_center": round(body_center, 1),
                "stability": round(stability, 1),
                "knee_rom": round(knee_rom, 1) if knee_rom is not None else None,
                "hip_rom": round(hip_rom, 1) if hip_rom is not None else None,
                "elbow_rom": round(elbow_rom, 1) if elbow_rom is not None else None,
                "arm_reach": round(arm_reach, 2) if arm_reach is not None else None,
                "elbow_finish": round(elbow_finish, 1) if elbow_finish is not None else None,
                "hand_high": round(hand_high, 2) if hand_high is not None else None,
            },
        }


def _mean(*values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)
