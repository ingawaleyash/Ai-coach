from pose_analyzer import PoseAnalyzer


model_path = r"D:\Project\AI-Cricket\analysis\ai\pose_landmarker_full.task"

video_path = r"D:\Project\AI-Cricket\media\videos\flip.mp4"

output_path = r"D:\Project\AI-Cricket\media\videos\pose_output.mp4"


analyzer = PoseAnalyzer(model_path)

analyzer.process_video(
    video_path,
    output_path
)