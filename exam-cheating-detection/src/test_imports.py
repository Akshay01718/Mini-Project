import traceback
import sys
import os

# Add parent directory to path to access config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    print("✓ cv2 imported")
    import yaml
    print("✓ yaml imported")
    from datetime import datetime
    print("✓ datetime imported")
    
    from detection.face_detection import FaceDetector
    print("✓ FaceDetector imported")
    from detection.eye_tracking import EyeTracker
    print("✓ EyeTracker imported")
    from detection.mouth_detection import MouthMonitor
    print("✓ MouthMonitor imported")
    from detection.object_detection import ObjectDetector
    print("✓ ObjectDetector imported")
    from detection.multi_face import MultiFaceDetector
    print("✓ MultiFaceDetector imported")
    
    from utils.video_utils import VideoRecorder
    print("✓ VideoRecorder imported")
    from utils.screen_capture import ScreenRecorder
    print("✓ ScreenRecorder imported")
    from utils.logging import AlertLogger
    print("✓ AlertLogger imported")
    from utils.alert_system import AlertSystem
    print("✓ AlertSystem imported")
    from utils.violation_logger import ViolationLogger
    print("✓ ViolationLogger imported")
    from utils.screenshot_utils import ViolationCapturer
    print("✓ ViolationCapturer imported")
    from reporting.report_generator import ReportGenerator
    print("✓ ReportGenerator imported")
    
    print("\nLoading config...")
    config_path = '../config/config.yaml' if os.path.exists('../config/config.yaml') else 'config/config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print("✓ Config loaded")
    
    print("\nInitializing modules...")
    alert_logger = AlertLogger(config)
    print("✓ AlertLogger initialized")
    alert_system = AlertSystem(config)
    print("✓ AlertSystem initialized")
    violation_capturer = ViolationCapturer(config)
    print("✓ ViolationCapturer initialized")
    violation_logger = ViolationLogger(config)
    print("✓ ViolationLogger initialized")
    report_generator = ReportGenerator(config)
    print("✓ ReportGenerator initialized")
    
    print("\n✓✓✓ All modules loaded successfully! ✓✓✓")
    
except Exception as e:
    print(f"\n✗ Error occurred:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
