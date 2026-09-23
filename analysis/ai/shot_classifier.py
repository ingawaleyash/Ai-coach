"""
Cricket shot classification — RULE-BASED BASELINE.

IMPORTANT (academic honesty):
This is NOT a trained machine-learning classifier. There is no
cricket-shot training dataset in this project, so no accuracy is
claimed. This module implements a transparent, explainable heuristic
that reads the batsman's arm swing plane and hand height from the pose
landmarks and matches them against simple cricket biomechanics.

If the evidence is weak, it returns "Unknown" rather than guessing.

The interface is deliberately small (`classify`), so a trained ML
classifier (e.g. an LSTM/transformer over the landmark time-series)
can later replace the internals without touching the pipeline.
"""

import statistics


# ----------------------------------------------------------------------
# Confidence floor — below this we refuse to name a shot.
# ----------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.40


def _safe_mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _safe_max(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def _find_swing_window(series):
    """
    Locate the swing phase: the contiguous slice where the (bat) arm
    changes angle the fastest. Returns (start, end) indices.
    """
    elbows = series.get("r_elbow") or series.get("l_elbow") or []
    if len(elbows) < 3:
        return (0, len(elbows))

    # absolute change between consecutive frames
    deltas = []
    for i in range(1, len(elbows)):
        if elbows[i] is not None and elbows[i - 1] is not None:
            deltas.append(abs(elbows[i] - elbows[i - 1]))
        else:
            deltas.append(0.0)

    if not deltas:
        return (0, len(elbows))

    # smooth with a tiny window to avoid picking a single noisy frame
    window = 3
    smoothed = []
    for i in range(len(deltas)):
        lo = max(0, i - window)
        hi = min(len(deltas), i + window + 1)
        smoothed.append(sum(deltas[lo:hi]) / (hi - lo))

    peak = max(range(len(smoothed)), key=lambda i: smoothed[i])
    # expand outward while activity stays above 40% of the peak
    threshold = smoothed[peak] * 0.4
    start = peak
    while start > 0 and smoothed[start] >= threshold:
        start -= 1
    end = peak
    while end < len(smoothed) - 1 and smoothed[end] >= threshold:
        end += 1
    return (start, end + 1)


class ShotClassifier:
    """
    Rule-based baseline classifier.

    `classify(series)` takes a dict of per-frame feature lists and
    returns:
        {
          "shot": str,
          "confidence": float (0-1),
          "reasoning": [str, ...],
          "swing": { "plane", "hand_height", "elbow_extension", "knee_bend" }
        }
    """

    def classify(self, series):
        reasoning = []

        # ---- default / empty guard ----
        if not series or not any(series.values()):
            return self._result("Unknown", 0.0, ["Not enough pose data."], None)

        # ---- locate the swing phase ----
        start, end = _find_swing_window(series)
        swing = {}
        if end <= start:
            swing["plane"] = None
            swing["hand_height"] = None
            swing["elbow_extension"] = None
            swing["knee_bend"] = None
        else:
            swing = self._swing_features(series, start, end)

        # ---- apply the rules ----
        shot, confidence, notes = self._apply_rules(swing)
        reasoning.extend(notes)

        if confidence < CONFIDENCE_THRESHOLD:
            shot = "Unknown"
            reasoning.append(
                "Confidence below threshold — not naming a shot."
            )

        reasoning.insert(
            0,
            "Rule-based baseline (no trained dataset): "
            "swing plane %.0f°, hand height %.2f, "
            "elbow extension %.0f°." % (
                swing["plane"] if swing["plane"] is not None else -1,
                swing["hand_height"] if swing["hand_height"] is not None else -1,
                swing["elbow_extension"] if swing["elbow_extension"] is not None else -1,
            ),
        )

        return self._result(shot, round(confidence, 3), reasoning, swing)

    # ------------------------------------------------------------------
    def _swing_features(self, series, start, end):
        """Compute swing-window discriminators from the raw series."""
        elbows = series.get("r_elbow") or series.get("l_elbow") or []
        knees = series.get("r_knee") or series.get("l_knee") or []
        hands = series.get("wrist_height") or []

        def slice_of(lst, key):
            out = []
            for i in range(start, min(end, len(lst))):
                v = lst[i]
                if v is not None:
                    out.append(v)
            return out

        swing_elbows = slice_of(elbows, "elbow")
        swing_knees = slice_of(knees, "knee")
        swing_hands = slice_of(hands, "hand")

        # Swing plane: angle of the (shoulder-mid -> wrist) line from
        # horizontal during the swing. Vertical bat ≈ steep angle.
        # Approximated from hand height vs arm extension.
        plane = None
        elbow_extension = _safe_max(swing_elbows)
        hand_height = _safe_max(swing_hands)
        if hand_height is not None:
            # 0.0 = hands at shoulder height; larger = hands above.
            # Roughly map to a 0-90° "bat verticality" scale.
            plane = min(90.0, max(0.0, 45.0 + hand_height * 40.0))

        knee_bend = None
        if swing_knees:
            # lower knee angle = more bent = deeper crouch
            knee_bend = 180.0 - _safe_min(swing_knees)

        return {
            "plane": plane,
            "hand_height": hand_height,
            "elbow_extension": elbow_extension,
            "knee_bend": knee_bend,
        }

    # ------------------------------------------------------------------
    def _apply_rules(self, swing):
        """
        Transparent rule table. Returns (shot, confidence, notes).
        """
        plane = swing.get("plane")
        hand_height = swing.get("hand_height")
        elbow_ext = swing.get("elbow_extension")
        knee_bend = swing.get("knee_bend")
        notes = []

        if plane is None:
            return "Unknown", 0.0, ["No reliable swing detected."]

        # ---- VERTICAL BAT family (steep swing plane) ----
        if plane >= 55:
            # Hands high above the shoulders → lofted / attacking
            if hand_height is not None and hand_height >= 1.0:
                conf = 0.55 + min(0.25, max(0.0, (plane - 55) / 140))
                notes.append(
                    "Steep bat + hands high over the shoulders → "
                    "lofted/attacking swing."
                )
                return "Lofted / Attacking Shot", conf, notes

            # Deep front-knee bend → front-foot drive / defence
            if knee_bend is not None and knee_bend >= 60:
                notes.append(
                    "Vertical bat with a deep front-knee crouch → "
                    "front-foot drive."
                )
                return "Straight / Cover Drive", 0.55, notes

            notes.append(
                "Vertical bat, hands around shoulder height → "
                "front-foot drive."
            )
            return "Straight / Cover Drive", 0.48, notes

        # ---- HORIZONTAL BAT family (shallow swing plane) ----
        if plane <= 35:
            # Strong horizontal arm extension → cut or pull
            if elbow_ext is not None and elbow_ext >= 150:
                notes.append(
                    "Shallow/horizontal bat with extended arms → "
                    "cut or pull (back-foot)."
                )
                return "Cut / Pull", 0.52, notes

            notes.append(
                "Shallow bat, hands around waist height → "
                "back-foot shot."
            )
            return "Back-foot shot", 0.44, notes

        # ---- BETWEEN (30-55°) ----
        if knee_bend is not None and knee_bend >= 60:
            notes.append("Moderate bat angle with a deep crouch.")
            return "Front-foot Defence", 0.42, notes

        notes.append("Ambiguous bat angle — evidence is weak.")
        return "Unknown", 0.30, notes

    # ------------------------------------------------------------------
    def _result(self, shot, confidence, reasoning, swing):
        return {
            "shot": shot,
            "confidence": confidence,
            "reasoning": reasoning,
            "swing": swing or {},
        }


def _safe_min(values):
    values = [v for v in values if v is not None]
    return min(values) if values else None
