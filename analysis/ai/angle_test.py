import cv2
import mediapipe as mp
import math


def calculate_angle(a, b, c):
    """Calculate angle ABC."""

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


model_path = r"D:\Project\AI-Cricket\analysis\ai\pose_landmarker_full.task"
video_path = r"D:\Project\AI-Cricket\media\videos\flip.mp4"


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


landmarker = PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


frame_count = 0


while True:

    success, frame = cap.read()

    if not success:
        break

    frame_count += 1

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms = int(
        (frame_count / fps) * 1000
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]


        # MediaPipe landmark numbers
        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]

        left_hip = landmarks[23]
        left_knee = landmarks[25]
        left_ankle = landmarks[27]


        # Convert normalized coordinates
        shoulder = (
            left_shoulder.x,
            left_shoulder.y
        )

        elbow = (
            left_elbow.x,
            left_elbow.y
        )

        wrist = (
            left_wrist.x,
            left_wrist.y
        )

        hip = (
            left_hip.x,
            left_hip.y
        )

        knee = (
            left_knee.x,
            left_knee.y
        )

        ankle = (
            left_ankle.x,
            left_ankle.y
        )


       # Right-side landmarks
right_shoulder = landmarks[12]
right_elbow = landmarks[14]
right_wrist = landmarks[16]

right_hip = landmarks[24]
right_knee = landmarks[26]
right_ankle = landmarks[28]


# Convert right-side coordinates
r_shoulder = (
    right_shoulder.x,
    right_shoulder.y
)

r_elbow = (
    right_elbow.x,
    right_elbow.y
)

r_wrist = (
    right_wrist.x,
    right_wrist.y
)

r_hip = (
    right_hip.x,
    right_hip.y
)

r_knee = (
    right_knee.x,
    right_knee.y
)

r_ankle = (
    right_ankle.x,
    right_ankle.y
)


# Calculate right-side angles
right_elbow_angle = calculate_angle(
    r_shoulder,
    r_elbow,
    r_wrist
)

right_knee_angle = calculate_angle(
    r_hip,
    r_knee,
    r_ankle
)


# Hip angle
left_hip_angle = calculate_angle(
    shoulder,
    hip,
    knee
)

right_hip_angle = calculate_angle(
    r_shoulder,
    r_hip,
    r_knee
)


print(
    f"Frame {frame_count}: "
    f"L-Elbow={elbow_angle:.1f}° | "
    f"R-Elbow={right_elbow_angle:.1f}° | "
    f"L-Knee={knee_angle:.1f}° | "
    f"R-Knee={right_knee_angle:.1f}° | "
    f"L-Hip={left_hip_angle:.1f}° | "
    f"R-Hip={right_hip_angle:.1f}°"
)

print(
    f"Frame {frame_count}: "
    f"Elbow = {elbow_angle:.1f}° | "
    f"Knee = {knee_angle:.1f}°"
)


cap.release()

landmarker.close()

print()
print("--------------------------------")
print("ANGLE ANALYSIS COMPLETE")
print("--------------------------------")
print("Frames analyzed:", frame_count)