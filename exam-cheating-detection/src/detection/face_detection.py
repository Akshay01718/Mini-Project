import cv2
import numpy as np
from datetime import datetime

class FaceDetector:
    def __init__(self, config):
        # Use OpenCV's Haar Cascade for face detection (built-in, no external dependencies)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.config = config
        self.detection_interval = config['detection']['face']['detection_interval']
        self.min_confidence = config['detection']['face']['min_confidence']
        self.frame_count = 0
        self.face_present = False
        self.last_face_time = None
        self.alert_logger = None
        self.face_disappeared_start = None

    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger

    def detect_face(self, frame):
        self.frame_count += 1
        if self.frame_count % self.detection_interval != 0:
            return self.face_present
            
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More thorough search (was 1.1)
            minNeighbors=3,    # Less strict (was 5) to detection faces easier
            minSize=(30, 30)
        )
        
        current_time = datetime.now()
        if len(faces) > 0:
            if not self.face_present and self.face_disappeared_start:
                disappearance_duration = (current_time - self.face_disappeared_start).total_seconds()
                if disappearance_duration > 5 and self.alert_logger:
                    self.alert_logger.log_alert(
                        "FACE_REAPPEARED",
                        f"Face reappeared after {disappearance_duration:.1f} seconds"
                    )
            
            self.face_present = True
            self.last_face_time = current_time
            self.face_disappeared_start = None
            return True
        else:
            if self.face_present:
                self.face_disappeared_start = current_time
                
            self.face_present = False
            if self.last_face_time and (current_time - self.last_face_time).total_seconds() > 5:
                if self.alert_logger:
                    self.alert_logger.log_alert(
                        "FACE_DISAPPEARED",
                        "Face disappeared for more than 5 seconds"
                    )
            return False