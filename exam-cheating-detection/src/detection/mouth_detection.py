import cv2
import numpy as np

class MouthMonitor:
    def __init__(self, config):
        # Use OpenCV's Haar Cascade for face and mouth detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Note: There's no built-in mouth cascade, so we'll use a simpler approach
        # We'll detect face and estimate mouth region, then check for movement using frame differencing
        
        self.mouth_threshold = config['detection']['mouth']['movement_threshold']
        self.mouth_movement_count = 0
        self.last_mouth_time = None
        self.alert_logger = None
        self.prev_mouth_region = None
        
    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger
        
    def monitor_mouth(self, frame):
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                self.prev_mouth_region = None
                return False
            
            # Get the first face
            (x, y, w, h) = faces[0]
            
            # Estimate mouth region (lower third of face)
            mouth_y = y + int(h * 0.6)
            mouth_h = int(h * 0.4)
            mouth_x = x + int(w * 0.25)
            mouth_w = int(w * 0.5)
            
            # Extract mouth region
            mouth_region = gray[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w]
            
            if mouth_region.size == 0:
                return False
            
            # Check for movement by comparing with previous frame
            if self.prev_mouth_region is not None and self.prev_mouth_region.shape == mouth_region.shape:
                # Calculate difference between current and previous mouth region
                diff = cv2.absdiff(mouth_region, self.prev_mouth_region)
                movement = np.mean(diff)
                
                # If significant movement detected (lowered threshold for better detection)
                if movement > 5:  # Threshold for mouth movement (lowered from 10 to 5)
                    self.mouth_movement_count += 1
                    
                    if self.mouth_movement_count > self.mouth_threshold and self.alert_logger:
                        self.alert_logger.log_alert(
                            "MOUTH_MOVEMENT", 
                            "Excessive mouth movement detected (possible talking)"
                        )
                        self.mouth_movement_count = 0
                    
                    self.prev_mouth_region = mouth_region.copy()
                    return True
                else:
                    self.mouth_movement_count = max(0, self.mouth_movement_count - 1)
            
            self.prev_mouth_region = mouth_region.copy()
            return False
            
        except Exception as e:
            # Silently handle errors and return False
            return False