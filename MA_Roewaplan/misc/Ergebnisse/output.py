import cv2
from ultralytics import YOLO

# Load the Yolov8 model
model = YOLO('yolov8n-obb.pt')

# Open the video file
video_path = 'https://www.youtube.com/watch?v=Vz4f8Gy6P1Q'
cap = cv2.VideoCapture(video_path)

# Loop trough the video frames
while cap.isOpened():
    # Read the frame of the video
    success, frame = cap.read()

    if success:
        # Rund Yolov8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow('YoloV8 Inference', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            # Break the loop is the video is finished
            break

cap.release()
cv2.destroyAllWindows()