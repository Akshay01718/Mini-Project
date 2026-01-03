import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

class UnifiedFaceDetector:
    def __init__(self, config):
        self.config = config
        
        # Initialize MediaPipe Face Mesh
        try:
            from mediapipe import solutions
        except ImportError:
            import mediapipe.python.solutions as solutions
            
        self.mp_face_mesh = solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Crucial for iris tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.alert_logger = None
        
        # State tracking
        self.face_present = False
        self.last_face_time = None
        self.face_disappeared_start = None
        
        self.gaze_direction = "Center"
        self.last_gaze_change = datetime.now()
        self.gaze_changes = 0
        
        # EAR/MAR parameters
        # GAP/MAR parameters
        # Mouth Aspect Ratio thresholds
        self.MAR_THRESHOLD = 0.15  # MUCH Lower threshold for subtle talking
        self.mouth_state_history = []
        self.MOUTH_HISTORY_LEN = 20 # Longer history to catch slow speech
        
    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger

    def _get_aspect_ratio(self, p1, p2, p3, p4, p5, p6):
        """Calculate aspect ratio for eyes or mouth"""
        # Vertical distances
        A = np.linalg.norm(np.array(p2) - np.array(p6))
        B = np.linalg.norm(np.array(p3) - np.array(p5))
        # Horizontal distance
        C = np.linalg.norm(np.array(p1) - np.array(p4))
        if C == 0: return 0
        return (A + B) / (2.0 * C)

    def _get_iris_position(self, iris_center, eye_corners):
        """Calculate normalized iris position (0.0=left, 1.0=right)"""
        center_x = iris_center[0]
        left_x = eye_corners[0][0]
        right_x = eye_corners[1][0]
        
        width = right_x - left_x
        if width == 0: return 0.5
        
        # Normalized position: 0 (left) to 1 (right)
        return (center_x - left_x) / width

    def process_frame(self, frame):
        """
        Process frame and return all detection results in a single pass.
        Returns: params dict with face_present, gaze_direction, eye_ratio, mouth_moving
        """
        results_dict = {
            'face_present': False,
            'gaze_direction': 'Unknown',
            'eye_ratio': 0.0,
            'mouth_moving': False,
            'mouth_open_score': 0.0
        }
        
        # RGB Conversion for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        current_time = datetime.now()
        
        if not results.multi_face_landmarks:
            # Handle Face Disappearance Logic
            if self.face_present:
                self.face_disappeared_start = current_time
            self.face_present = False
            
            if self.face_disappeared_start and (current_time - self.face_disappeared_start).total_seconds() > 5:
                if self.alert_logger:
                    self.alert_logger.log_alert("FACE_DISAPPEARED", "Face disappeared for > 5s")
                    # Reset timer to avoid spam
                    self.face_disappeared_start = current_time 
            
            return results_dict

        # Face is present
        self.face_present = True
        self.face_disappeared_start = None
        results_dict['face_present'] = True
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        
        # ---------------------------------------------------------
        # 1. Gaze Tracking (Using Iris Landmarks)
        # ---------------------------------------------------------
        # Right Eye (User's Left): 474, 476, 475, 477 (Iris), 33 (Corner1), 133 (Corner2)
        # Left Eye (User's Right): 469, 471, 470, 472 (Iris), 362 (Corner1), 263 (Corner2)
        
        # Get coordinates for Left Eye (User's Right)
        l_iris = np.array([landmarks[473].x, landmarks[473].y]) # Center of left iris
        l_corner_in = np.array([landmarks[362].x, landmarks[362].y])
        l_corner_out = np.array([landmarks[263].x, landmarks[263].y])
        
        # Get coordinates for Right Eye (User's Left)
        r_iris = np.array([landmarks[468].x, landmarks[468].y]) # Center of right iris
        r_corner_in = np.array([landmarks[133].x, landmarks[133].y])
        r_corner_out = np.array([landmarks[33].x, landmarks[33].y])
        
        # Calculate horizontal ratios
        # Note: Mirror effect - if iris is closer to "out" corner, they are looking the other way
        gaze_ratio_l = self._get_iris_position(l_iris, [l_corner_out, l_corner_in])
        gaze_ratio_r = self._get_iris_position(r_iris, [r_corner_out, r_corner_in])
        avg_gaze_ratio = (gaze_ratio_l + gaze_ratio_r) / 2.0
        
        # Determine Direction
        # Range is 0.0 (Left-most) to 1.0 (Right-most). Center is ~0.5.
        # NARROWER THRESHOLDS for better sensitivity at distance
        new_gaze = "Center"
        if avg_gaze_ratio < 0.46:  
            new_gaze = "Right" # Mirrored: User looking to their Right (Screen Left)
        elif avg_gaze_ratio > 0.54:
            new_gaze = "Left" # Mirrored: User looking to their Left (Screen Right)
            
        results_dict['gaze_direction'] = new_gaze
        
        # Log Gaze Alerts
        if new_gaze != self.gaze_direction:
            self.gaze_changes += 1
            self.gaze_direction = new_gaze
            self.last_gaze_change = current_time
            
        if self.gaze_changes > 5 and (current_time - self.last_gaze_change).total_seconds() < 2:
             if self.alert_logger:
                self.alert_logger.log_alert("GAZE_MOVEMENT", "Rapid gaze shifting detected")
             self.gaze_changes = 0

        # ---------------------------------------------------------
        # 2. Mouth Tracking (Mouth Aspect Ratio - MAR)
        # ---------------------------------------------------------
        # Mouth Landmarks:
        # Top: 13, Bottom: 14
        # Left: 61, Right: 291
        # Inner lips for better "open" detection
        
        mouth_top = (landmarks[13].x * w, landmarks[13].y * h)
        mouth_bottom = (landmarks[14].x * w, landmarks[14].y * h)
        mouth_left = (landmarks[61].x * w, landmarks[61].y * h)
        mouth_right = (landmarks[291].x * w, landmarks[291].y * h)
        
        # Vertical / Horizontal
        mouth_height = np.linalg.norm(np.array(mouth_top) - np.array(mouth_bottom))
        mouth_width = np.linalg.norm(np.array(mouth_left) - np.array(mouth_right))
        
        if mouth_width == 0: mouth_width = 1
        mar = mouth_height / mouth_width
        results_dict['mouth_open_score'] = mar
        
        is_mouth_open = mar > self.MAR_THRESHOLD
        
        # Simple temporal smoothing for mouth moving (talking)
        self.mouth_state_history.append(1 if is_mouth_open else 0)
        if len(self.mouth_state_history) > self.MOUTH_HISTORY_LEN:
            self.mouth_state_history.pop(0)
            
        # If mouth has been open and closed frequently, it's talking
        activity = sum(self.mouth_state_history)
        # Lower activity requirement: even 2 frames of "open" in the history window triggers alert
        if activity >= 2: 
             results_dict['mouth_moving'] = True
             if self.alert_logger and activity == 2: # Log once when starting
                 self.alert_logger.log_alert("MOUTH_MOVING", "Talking detected (Mouth Aspect Ratio)")
        else:
             results_dict['mouth_moving'] = False
             
        # ---------------------------------------------------------
        # 3. Eye Aspect Ratio (EAR) for Drowsiness/Blink
        # ---------------------------------------------------------
        # Simple proxy: distance between upper and lower lid
        # Left Eye Lids: 159 (top), 145 (bottom)
        l_top = np.array([landmarks[159].x, landmarks[159].y])
        l_bot = np.array([landmarks[145].x, landmarks[145].y])
        ear = np.linalg.norm(l_top - l_bot) * 10 # Scale up
        results_dict['eye_ratio'] = ear

        return results_dict
