import cv2
import numpy as np
from predictor import predict_from_rgb

class Camera:
    def __init__(self, roi_size=20):
        # Start video capture from default camera
        self.video = cv2.VideoCapture(0)
        self.roi_size = roi_size
        self.last_rgb = (0, 0, 0)  # Initialize with black

    def __del__(self):
        # Release camera resource on delete
        self.video.release()

    def get_last_rgb(self):
        # Return last RGB values (for optional API call)
        return self.last_rgb

    def get_frame(self):
        # Read frame from webcam
        success, frame = self.video.read()
        if not success:
            return b''

        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape
        cx, cy = width // 2, height // 2
        half = self.roi_size // 2

        # Define ROI square in center
        roi = frame[cy - half:cy + half, cx - half:cx + half]

        # Calculate mean color in ROI
        mean_bgr = cv2.mean(roi)[:3]
        mean_rgb = tuple(int(c) for c in mean_bgr[::-1])  # Convert BGR to RGB
        self.last_rgb = mean_rgb

        # Predict using loaded ML model
        label = predict_from_rgb(*mean_rgb)

        # Draw ROI box
        cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half), (0, 255, 0), 2)

        # Draw label
        text = f"{label} ({mean_rgb})"
        cv2.putText(frame, text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Encode to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
