"""
Top-level AI analysis pipeline.

Coordinates every stage for a single uploaded video:

    read video → pose detect → feature extract → classify shot
              → score technique → generate feedback → write annotated video

Returns a plain dict that the Django layer stores in an AnalysisResult
row. The pipeline is synchronous (fine for the first working version)
and is deliberately structured so it can later be run in a background
worker without changing its internals.
"""

import os

from .pose_analyzer import PoseAnalyzer
from .features import FeatureExtractor, FeatureAggregator
from .shot_classifier import ShotClassifier
from .technique_scorer import TechniqueScorer
from .feedback_generator import generate_feedback

# Path to the MediaPipe pose model (kept next to this package).
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")


class AnalysisError(Exception):
    """Raised when a video cannot be analysed for a user-facing reason."""


def run_pipeline(video_path, output_path=None):
    """
    Analyse `video_path`, optionally writing an annotated video to
    `output_path`. Returns a result dict ready for storage.

    Raises AnalysisError for user-facing failures.
    """
    if not os.path.exists(MODEL_PATH):
        raise AnalysisError("Pose model file is missing on the server.")

    if not os.path.exists(video_path):
        raise AnalysisError("Uploaded video could not be found on the server.")

    analyzer = PoseAnalyzer(MODEL_PATH)
    extractor = None
    aggregator = FeatureAggregator()
    last_features = {}

    def on_frame(landmarks, frame_index, width, height):
        nonlocal extractor, last_features
        if landmarks is None:
            return None
        if extractor is None:
            extractor = FeatureExtractor(width, height)
        feats = extractor.extract(landmarks)
        if feats:
            aggregator.add(feats)
            last_features = feats
            # overlay live measurements on the annotated frame
            overlay = {}
            if feats.get("l_elbow") is not None:
                overlay["L-Elbow"] = f"{feats['l_elbow']:.0f}°"
            if feats.get("r_elbow") is not None:
                overlay["R-Elbow"] = f"{feats['r_elbow']:.0f}°"
            if feats.get("l_knee") is not None:
                overlay["L-Knee"] = f"{feats['l_knee']:.0f}°"
            if feats.get("torso_angle") is not None:
                overlay["Torso"] = f"{feats['torso_angle']:.0f}°"
            return overlay
        return None

    frame_stats = analyzer.analyze(
        video_path,
        output_path=output_path,
        on_frame=on_frame,
    )

    if frame_stats["total_frames"] == 0:
        raise AnalysisError("The video has no readable frames.")

    if frame_stats["detected_frames"] == 0:
        raise AnalysisError(
            "No person / pose was detected in the video. "
            "Please upload a video of a batter in frame."
        )

    # ----- summarise features -----
    summary = aggregator.summarize()

    # ----- classify shot -----
    series = aggregator.series()
    classifier = ShotClassifier()
    classification = classifier.classify(series)

    # ----- score technique -----
    scorer = TechniqueScorer()
    scores = scorer.score(summary)

    # ----- feedback -----
    observations, feedback = generate_feedback(scores, summary)

    # ----- measurements dict for storage / display -----
    measurements = {
        "summary": summary,
        "swing": classification.get("swing", {}),
        "score_raw": scores.get("_raw", {}),
    }

    return {
        "total_frames": frame_stats["total_frames"],
        "detected_frames": frame_stats["detected_frames"],
        "detection_percentage": frame_stats["detection_percentage"],
        "detected_shot": classification["shot"],
        "shot_confidence": classification["confidence"],
        "shot_reasoning": classification.get("reasoning", []),
        "scores": scores,
        "measurements": measurements,
        "observations": observations,
        "feedback": feedback,
    }
