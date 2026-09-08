import cv2


def read_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # Display the current frame
        cv2.imshow("AI Coach Cricketer - Video", frame)

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Total frames processed:", frame_count)