import cv2
import mediapipe as mp


class PoseAnalyzer:

    def __init__(self, model_path):

        self.model_path = model_path

        # MediaPipe Tasks API
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.RunningMode = mp.tasks.vision.RunningMode

        # Configure Pose Landmarker for VIDEO mode
        options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=self.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = self.PoseLandmarker.create_from_options(
            options
        )

    def process_video(self, video_path, output_path=None):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("ERROR: Could not open video.")
            return

        frame_count = 0
        detected_frames = 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        writer = None

        if output_path:

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            # Convert OpenCV BGR → RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Convert frame to MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Timestamp in milliseconds
            timestamp_ms = int(
                (frame_count / fps) * 1000
            )

            # Run pose detection
            result = self.landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # Check if a person was detected
            if result.pose_landmarks:

                detected_frames += 1

                # Draw landmarks
                for pose_landmarks in result.pose_landmarks:

                    for landmark in pose_landmarks:

                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        # Keep points inside frame
                        if (
                            0 <= x < width
                            and 0 <= y < height
                        ):

                            cv2.circle(
                                frame,
                                (x, y),
                                4,
                                (0, 255, 0),
                                -1
                            )

                    # Draw connections
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 7),
                        (0, 4), (4, 5), (5, 6), (6, 8),

                        (9, 10),

                        (11, 12),

                        (11, 13), (13, 15),
                        (15, 17), (15, 19),
                        (15, 21),

                        (12, 14), (14, 16),
                        (16, 18), (16, 20),
                        (16, 22),

                        (11, 23),
                        (12, 24),

                        (23, 24),

                        (23, 25),
                        (25, 27),
                        (27, 29),
                        (29, 31),

                        (24, 26),
                        (26, 28),
                        (28, 30),
                        (30, 32)
                    ]

                    for start, end in connections:

                        start_point = pose_landmarks[start]
                        end_point = pose_landmarks[end]

                        x1 = int(start_point.x * width)
                        y1 = int(start_point.y * height)

                        x2 = int(end_point.x * width)
                        y2 = int(end_point.y * height)

                        cv2.line(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                cv2.putText(
                    frame,
                    "POSE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "NO POSE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            # Write processed video
            if writer:
                writer.write(frame)

            # Display
            cv2.imshow(
                "AI Coach Cricketer - Pose Analysis",
                frame
            )

            # Press Q to stop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()

        if writer:
            writer.release()

        self.landmarker.close()

        cv2.destroyAllWindows()

        print()
        print("--------------------------------")
        print("AI POSE ANALYSIS COMPLETE")
        print("--------------------------------")
        print("Total frames:", frame_count)
        print("Frames with pose:", detected_frames)

        if frame_count > 0:

            percentage = (
                detected_frames / frame_count
            ) * 100

            print(
                f"Pose detection rate: {percentage:.2f}%"
            )