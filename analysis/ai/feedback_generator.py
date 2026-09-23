"""
Coaching feedback generator.

Produces human-readable coaching insights from the measured technique
scores and aggregated feature statistics. Every suggestion is grounded
in an actual measurement — nothing is fabricated or generic.

All messages carry an explicit disclaimer that this is AI-generated
guidance and not a substitute for a qualified coach.
"""

_DISCLAIMER = (
    "AI-generated coaching guidance. "
    "This is not a substitute for a qualified cricket coach."
)


def generate_feedback(scores, summary):
    """
    Return (key_observations, coaching_feedback):

      key_observations — short ✓/⚠ bullet points for the results page.
      coaching_feedback — structured dicts with
          {issue, why, suggestion, severity}.

    Both are driven entirely by measured data.
    """
    observations = []
    feedback = []

    # helper: add an observation
    def obs(icon, text):
        observations.append({"icon": icon, "text": text})

    def fb(issue, why, suggestion, severity="info"):
        feedback.append({
            "issue": issue,
            "why": why,
            "suggestion": suggestion,
            "severity": severity,
        })

    # --- Posture ---
    torso = _stat_mean(summary, "torso_angle")
    head_x = _stat_mean(summary, "head_x_offset")
    head_h = _stat_mean(summary, "head_height_rel")

    if torso is not None and abs(torso) <= 8:
        obs("✓", "Good upright torso posture.")
    elif torso is not None:
        direction = "leans right" if torso > 0 else "leans left"
        obs("⚠", f"Torso {direction} (avg {torso:.1f}°).")
        fb(
            "Torso lean",
            f"The torso averaged a {abs(torso):.1f}° "
            f"{'right' if torso > 0 else 'left'} lean.",
            "Work on keeping your torso closer to vertical, "
            "especially during the stroke, for better balance.",
            severity="warning",
        )

    if head_x is not None and abs(head_x) <= 0.15:
        obs("✓", "Head stays over the body.")
    elif head_x is not None:
        obs("⚠", "Head drifts off the body's centre-line.")
        fb(
            "Head alignment",
            "The head moved sideways from the body's midline. "
            "A stable head helps with balance and shot timing.",
            "Focus on keeping your head level and centred "
            "over your hips through the stroke.",
            severity="warning",
        )

    if head_h is not None and 0.8 <= head_h <= 1.6:
        obs("✓", "Head height is stable and natural.")
    elif head_h is not None:
        obs("⚠", "Head height appears unusually low or high.")

    # --- Balance ---
    balance_off = _stat_mean(summary, "balance_offset")
    stance = _stat_mean(summary, "stance_width")

    if balance_off is not None and abs(balance_off) <= 0.15:
        obs("✓", "Good balance over the support base.")
    elif balance_off is not None:
        obs("⚠", "Balance shifts to one side.")
        fb(
            "Lateral balance",
            f"The body's centre was offset by {abs(balance_off):.2f}×torso "
            f"{'to the' if balance_off > 0 else 'from the'} support base.",
            "Practice maintaining weight equally between "
            "your feet in your batting stance.",
            severity="warning",
        )

    if stance is not None and 0.7 <= stance <= 1.4:
        obs("✓", "Stance width looks stable.")
    elif stance is not None:
        obs("⚠", f"Stance width ({stance:.2f}×torso) is outside the typical range.")

    # --- Lower body ---
    knee_rom = _rom(summary, "l_knee")
    hip_rom = _rom(summary, "l_hip")

    if knee_rom is not None and knee_rom >= 30:
        obs("✓", "Good lower-body movement / knee bend.")
    elif knee_rom is not None:
        obs("⚠", "Limited knee movement during the stroke.")
        fb(
            "Knee movement",
            f"Knee range of motion was only {knee_rom:.1f}°.",
            "Bending the front knee into the shot can improve "
            "stability and power transfer.",
            severity="warning",
        )

    if hip_rom is not None and hip_rom >= 25:
        obs("✓", "Good hip rotation through the stroke.")

    # --- Upper body ---
    elbow_rom = _rom(summary, "l_elbow")
    reach_max = _arm_reach_max(summary)

    if elbow_rom is not None and elbow_rom >= 80:
        obs("✓", "Solid arm extension through the swing.")
    elif elbow_rom is not None:
        obs("⚠", "Arms didn't fully extend during the swing.")
        fb(
            "Arm extension",
            f"The elbow range of motion was {elbow_rom:.1f}°.",
            "Extending the arms through the contact zone "
            "can add power and control to your shots.",
            severity="warning",
        )

    if reach_max is not None and reach_max >= 1.0:
        obs("✓", "Good arm reach / follow-through.")

    # --- Follow through ---
    elbow_finish = _mean_optional(
        _stat_mean(summary, "l_elbow", "max"),
        _stat_mean(summary, "r_elbow", "max"),
    )
    hand_high = _stat_mean(summary, "wrist_height", "max")

    if elbow_finish is not None and elbow_finish >= 150:
        obs("✓", "Arm finishes in an extended follow-through.")
    elif elbow_finish is not None:
        obs("⚠", "Follow-through is short / arms remain bent.")
        fb(
            "Follow-through",
            f"Max elbow extension was {elbow_finish:.1f}°.",
            "Finishing with a full, extended follow-through "
            "helps complete the stroke and transfer energy.",
            severity="info",
        )

    if hand_high is not None and hand_high >= 0.9:
        obs("✓", "Hands finish high — good for lofted/attacking play.")

    # --- Shot-related tips (from classifier reasoning) ---
    # The pipeline passes the shot string; we don't add feedback here,
    # but the classifier's reasoning is displayed separately on the
    # results page.

    # Ensure at least one observation
    if not observations:
        obs("✓", "Pose was detected — analysis complete.")
        obs("ℹ", "Add more video or higher-resolution footage for richer insights.")

    # Add disclaimer
    fb("Disclaimer", _DISCLAIMER, "", severity="info")

    return observations, feedback


# ------------------------------------------------------------------
# Tiny helpers
# ------------------------------------------------------------------
def _stat_mean(summary, feature, stat="mean"):
    try:
        return float(summary[feature][stat])
    except (KeyError, TypeError, ValueError):
        return None


def _rom(summary, feature):
    return _stat_mean(summary, feature, "range")


def _arm_reach_max(summary):
    l = _stat_mean(summary, "l_arm_reach", "max")
    r = _stat_mean(summary, "r_arm_reach", "max")
    return _mean_optional(l, r)


def _mean_optional(a, b):
    vals = [v for v in (a, b) if v is not None]
    return sum(vals) / len(vals) if vals else None
