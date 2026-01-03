import cv2
import numpy as np
from datetime import datetime

class EyeTracker:
    def __init__(self, config):
        # Use OpenCV's Haar Cascade for eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        self.config = config
        self.eye_threshold = config['detection']['eyes']['gaze_threshold']
        self.last_gaze_change = datetime.now()
        self.gaze_direction = "Center"  # Default value
        self.eye_ratio = 0.3  # Default open eye ratio
        self.gaze_changes = 0
        self.alert_logger = None
        
        # For EAR (Eye Aspect Ratio) calculation
        self.EYE_ASPECT_RATIO_THRESH = 0.3
        self.EYE_ASPECT_RATIO_CONSEC_FRAMES = 3

    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger

    def track_eyes(self, frame):
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces first
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return self.gaze_direction, self.eye_ratio  # Return last known values
            
            # Get the first face
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes within the face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            
            if len(eyes) >= 2:
                # Eyes detected - assume open
                self.eye_ratio = 0.35
                
                # Simple gaze estimation based on eye positions
                # Sort eyes by x-coordinate to get left and right
                eyes_sorted = sorted(eyes, key=lambda e: e[0])
                left_eye = eyes_sorted[0]
                right_eye = eyes_sorted[1] if len(eyes_sorted) > 1 else left_eye
                
                # Calculate eye center positions relative to face
                left_eye_center_x = left_eye[0] + left_eye[2] // 2
                right_eye_center_x = right_eye[0] + right_eye[2] // 2
                eyes_center_x = (left_eye_center_x + right_eye_center_x) // 2
                face_center_x = w // 2
                
                # Determine gaze direction based on eye position relative to face center
                offset = eyes_center_x - face_center_x
                new_gaze = "Center"
                if offset < -w * 0.1:  # Looking left
                    new_gaze = "Left"
                elif offset > w * 0.1:  # Looking right
                    new_gaze = "Right"
                
                # Update gaze changes
                current_time = datetime.now()
                if new_gaze != self.gaze_direction:
                    self.gaze_changes += 1
                    self.gaze_direction = new_gaze
                    self.last_gaze_change = current_time
                    
                # Check for excessive eye movement
                if (self.gaze_changes > 3 and 
                    (current_time - self.last_gaze_change).total_seconds() < 2 and
                    self.alert_logger):
                    self.alert_logger.log_alert(
                        "EYE_MOVEMENT",
                        "Excessive eye movement detected"
                    )
                    self.gaze_changes = 0
            else:
                # No eyes detected or only one eye - might be closed or looking away
                self.eye_ratio = 0.2
            
            return self.gaze_direction, self.eye_ratio
            
        except Exception as e:
            if self.alert_logger:
                self.alert_logger.log_alert(
                    "EYE_TRACKING_ERROR",
                    f"Error in eye tracking: {str(e)}"
                )
            return self.gaze_direction, self.eye_ratio  # Return last known values