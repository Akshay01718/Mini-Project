import cv2
import yaml
from datetime import datetime
from detection.face_detection import FaceDetector
from detection.eye_tracking import EyeTracker
from detection.mouth_detection import MouthMonitor
from detection.object_detection import ObjectDetector
from detection.multi_face import MultiFaceDetector
from utils.video_utils import VideoRecorder
from utils.screen_capture import ScreenRecorder
from utils.logging import AlertLogger
from utils.alert_system import AlertSystem
from utils.violation_logger import ViolationLogger
from utils.screenshot_utils import ViolationCapturer
from reporting.report_generator import ReportGenerator


def load_config():
    with open('config/config.yaml') as f:
        return yaml.safe_load(f)

def main():
    print("Loading config...")
    config = load_config()
    print("✓ Config loaded")
    
    print("Initializing alert system...")
    alert_logger = AlertLogger(config)
    alert_system = AlertSystem(config)
    violation_capturer = ViolationCapturer(config)
    violation_logger = ViolationLogger(config)
    report_generator = ReportGenerator(config)
    print("✓ Alert system initialized")
    
    print("Initializing recorders...")
    video_recorder = VideoRecorder(config)
    screen_recorder = ScreenRecorder(config)
    print("✓ Recorders initialized")
    
    print("Initializing detectors...")
    detectors = [
        FaceDetector(config),
        EyeTracker(config),
        MouthMonitor(config),
        MultiFaceDetector(config),
        ObjectDetector(config),
    ]
    print("✓ Detectors initialized")
    
    for detector in detectors:
        if hasattr(detector, 'set_alert_logger'):
            detector.set_alert_logger(alert_logger)
    print("✓ Alert loggers set")
    
    print("Starting webcam...")
    cap = cv2.VideoCapture(config['video']['source'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['video']['resolution'][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['video']['resolution'][1])
    print(f"✓ Webcam opened: {cap.isOpened()}")
    
    print("Reading first frame...")
    ret, frame = cap.read()
    print(f"✓ Frame read: {ret}, Shape: {frame.shape if ret else 'None'}")
    
    if ret:
        print("Testing detections...")
        try:
            face_present = detectors[0].detect_face(frame)
            print(f"✓ Face detection: {face_present}")
        except Exception as e:
            print(f"✗ Face detection error: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            gaze_direction, eye_ratio = detectors[1].track_eyes(frame)
            print(f"✓ Eye tracking: {gaze_direction}, {eye_ratio}")
        except Exception as e:
            print(f"✗ Eye tracking error: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            mouth_moving = detectors[2].monitor_mouth(frame)
            print(f"✓ Mouth detection: {mouth_moving}")
        except Exception as e:
            print(f"✗ Mouth detection error: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            multiple_faces = detectors[3].detect_multiple_faces(frame)
            print(f"✓ Multi-face detection: {multiple_faces}")
        except Exception as e:
            print(f"✗ Multi-face detection error: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            objects_detected = detectors[4].detect_objects(frame)
            print(f"✓ Object detection: {objects_detected}")
        except Exception as e:
            print(f"✗ Object detection error: {e}")
            import traceback
            traceback.print_exc()
    
    cap.release()
    print("\n✓✓✓ All tests passed! ✓✓✓")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
