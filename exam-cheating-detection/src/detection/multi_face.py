import cv2
import numpy as np

class MultiFaceDetector:
    def __init__(self, config):
        # Use OpenCV's Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.threshold = config['detection']['multi_face']['alert_threshold']
        self.consecutive_frames = 0
        self.alert_logger = None

    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger

    def detect_multiple_faces(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 1:
            # Count faces with reasonable confidence (all detected faces from Haar Cascade)
            high_conf_faces = len(faces)
            
            if high_conf_faces >= 2:
                self.consecutive_frames += 1
                if self.consecutive_frames >= self.threshold and self.alert_logger:
                    self.alert_logger.log_alert(
                        "MULTIPLE_FACES",
                        f"Detected {high_conf_faces} faces for {self.consecutive_frames} frames"
                    )
                    return True
        else:
            self.consecutive_frames = 0
            
        return False