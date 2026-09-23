"""
Headless MediaPipe pose analyzer.

Reads a video with OpenCV, runs MediaPipe PoseLandmarker in VIDEO mode,
draws the landmark skeleton + optional overlays, and optionally writes
an annotated output video. Unlike the earlier prototype it does not
open display windows or block on keyboard input, so it can run inside
the Django request/worker process.

The `on_frame` callback lets the caller (e.g. the feature extractor)
consume the raw landmarks frame by frame and return overlay text.
"""

import cv2
import mediapipe as mp


class PoseAnalyzer:

    def __init__(self, model_path):
        self.model_path = model_path

        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.RunningMode = mp.tasks.vision.RunningMode

        options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=self.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = self.PoseLandmarker.create_from_options(options)

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Skeleton connections (MediaPipe Pose topology)
    # ------------------------------------------------------------------
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12),
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        (11, 23), (12, 24),
        (23, 24),
        (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
    ]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def analyze(self, video_path, output_path=None, on_frame=None):
        """
        Process `video_path`.

        `on_frame(landmarks, frame_index, width, height)` is called for
        every frame. It may return a dict of {label: text} overlay lines
        to draw on the annotated frame, or None.

        Returns a dict with frame statistics.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open the video file: " + video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps is None:
            fps = 30.0

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_path, fourcc, fps, (width, height)
            )

        frame_count = 0
        detected_frames = 0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_count += 1

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )
                timestamp_ms = int((frame_count / fps) * 1000)

                result = self.landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )

                overlay_lines = None
                if result.pose_landmarks:
                    detected_frames += 1
                    landmarks = result.pose_landmarks[0]
                    self._draw_skeleton(frame, landmarks, width, height)

                    if on_frame:
                        overlay_lines = on_frame(
                            landmarks, frame_count, width, height
                        )

                    self._draw_status(frame, "POSE DETECTED", (0, 200, 0))
                else:
                    if on_frame:
                        overlay_lines = on_frame(
                            None, frame_count, width, height
                        )
                    self._draw_status(frame, "NO POSE", (0, 0, 220))

                if overlay_lines:
                    self._draw_overlay(frame, overlay_lines)

                if writer:
                    writer.write(frame)
        finally:
            cap.release()
            if writer:
                writer.release()
            self.close()

        detection_percentage = 0.0
        if frame_count > 0:
            detection_percentage = (detected_frames / frame_count) * 100

        return {
            "total_frames": frame_count,
            "detected_frames": detected_frames,
            "detection_percentage": round(detection_percentage, 2),
        }

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _draw_skeleton(self, frame, landmarks, width, height):
        # Landmark points
        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # Connections
        for start, end in self.CONNECTIONS:
            a = landmarks[start]
            b = landmarks[end]
            x1 = int(a.x * width)
            y1 = int(a.y * height)
            x2 = int(b.x * width)
            y2 = int(b.y * height)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def _draw_status(self, frame, text, color):
        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

    def _draw_overlay(self, frame, lines):
        """Draw overlay label/value lines in the top-left block."""
        y = 70
        for label, value in lines.items():
            cv2.putText(
                frame,
                f"{label}: {value}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 200, 0),
                2,
            )
            y += 28
