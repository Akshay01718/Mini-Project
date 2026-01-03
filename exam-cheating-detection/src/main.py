import cv2
import yaml
import os
from datetime import datetime
from detection.object_detection import ObjectDetector
from detection.multi_face import MultiFaceDetector
from detection.unified_detector import UnifiedFaceDetector  # [NEW] Robust MediaPipe Detector
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

def display_detection_results(frame, results):
    y_offset = 30
    line_height = 30
    
    # Status indicators
    status_items = [
        f"Face: {'Present' if results['face_present'] else 'Absent'}",
        f"Gaze: {results['gaze_direction']}",
        f"Eyes: {'Open' if results['eye_ratio'] > 0.15 else 'Closed'}", # Adjusted for distance (was 0.05)
        f"Mouth: {'Moving' if results['mouth_moving'] else 'Still'}"
    ]
    
    # Alert indicators
    alert_items = []
    if results['multiple_faces']:
        alert_items.append("Multiple Faces Detected!")
    if results['objects_detected']:
        alert_items.append("Suspicious Object Detected!")

    # Display status
    for item in status_items:
        cv2.putText(frame, item, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += line_height
    
    # Display alerts
    for item in alert_items:
        cv2.putText(frame, item, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += line_height
    
    # Timestamp
    cv2.putText(frame, results['timestamp'], 
               (frame.shape[1] - 250, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def main():
    config = load_config()
    alert_logger = AlertLogger(config)
    alert_system = AlertSystem(config)
    violation_capturer = ViolationCapturer(config)
    violation_logger = ViolationLogger(config)
    report_generator = ReportGenerator(config)

    student_info = {
        'id': 'STUDENT_001',
        'name': 'John Doe',
        'exam': 'Final Examination',
        'course': 'Computer Science 101'
    }

    print("\n" + "="*60)
    print("EXAM CHEATING DETECTION SYSTEM (MediaPipe Enhanced)")
    print("="*60)
    print("INSTRUCTIONS:")
    print("  - Press 'q' to stop the webcam and exit")
    print("  - Reports will be saved to: d:\\A1-mini\\exam-cheating-detection\\reports\\generated\\")
    print("  - Red boxes will appear around detected objects")
    print("="*60 + "\n")

    
    print("[DEBUG] Initializing recorders...")
    # Initialize recorders
    video_recorder = VideoRecorder(config)
    screen_recorder = ScreenRecorder(config)
    
    # Initialize audio monitor only if enabled
    audio_monitor = None
    if config['detection']['audio_monitoring']['enabled']:
        print("[DEBUG] Initializing audio monitor...")
        from detection.audio_detection import AudioMonitor
        audio_monitor = AudioMonitor(config)
        audio_monitor.alert_system = alert_system
        audio_monitor.alert_logger = alert_logger
        audio_monitor.start()

    cap = None  # Initialize to None so it can be safely accessed in finally block
    try:
        if config['screen']['recording']:
            print("[DEBUG] Starting screen recording...")
            screen_recorder.start_recording()
            
        print("[DEBUG] Initializing UnifiedDetector (MediaPipe)...")
        # Initialize detectors
        # [NEW] Single Unified Detector replaces Face/Eye/Mouth detectors
        # [NEW] Single Unified Detector replaces Face/Eye/Mouth detectors
        unified_detector = UnifiedFaceDetector(config)
        unified_detector.set_alert_logger(alert_logger)
        
        print("[DEBUG] Initializing Other Detectors...")
        # Keep specialized detectors
        multi_face_detector = MultiFaceDetector(config)
        object_detector = ObjectDetector(config)
        object_detector.set_alert_logger(alert_logger)

        print("[DEBUG] Starting Webcam...")
        # Start webcam recording
        video_recorder.start_recording()
        cap = cv2.VideoCapture(config['video']['source'])
        print(f"[DEBUG] Camera Opened: {cap.isOpened()}")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['video']['resolution'][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['video']['resolution'][1])
        
        print("[DEBUG] Entering Main Loop...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = {
                'face_present': False,
                'gaze_direction': 'Center',
                'eye_ratio': 0.3,
                'mouth_moving': False,
                'multiple_faces': False,
                'objects_detected': False,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # [NEW] Unified Detection Pass
            unified_results = unified_detector.process_frame(frame)
            results.update(unified_results)
            
            # [OLD] Legacy Detection Passes
            results['multiple_faces'] = multi_face_detector.detect_multiple_faces(frame)
            results['objects_detected'] = object_detector.detect_objects(frame, visualize=True)

            if not results['face_present']:
                violation_type = "FACE_DISAPPEARED"
                alert_system.speak_alert(violation_type)
                
                # Capture and log violation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                violation_image = violation_capturer.capture_violation(frame, violation_type, timestamp)
                violation_logger.log_violation(
                    violation_type,
                    timestamp,
                    {'duration': '5+ seconds', 'frame': results}
                )
                # alert_system.speak_alert("FACE_DISAPPEARED")
            elif results['multiple_faces']:
                violation_type = "MULTIPLE_FACES"
                alert_system.speak_alert(violation_type)
                
                # Capture and log violation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                violation_image = violation_capturer.capture_violation(frame, violation_type, timestamp)
                violation_logger.log_violation(
                    violation_type,
                    timestamp,
                    {'duration': '5+ seconds', 'frame': results}
                )
                # alert_system.speak_alert("MULTIPLE_FACES")
            elif results['objects_detected']:
                violation_type = "OBJECT_DETECTED"
                alert_system.speak_alert(violation_type)
                
                # Capture and log violation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                violation_image = violation_capturer.capture_violation(frame, violation_type, timestamp)
                violation_logger.log_violation(
                    violation_type,
                    timestamp,
                    {'duration': '5+ seconds', 'frame': results}
                )
                # alert_system.speak_alert("OBJECT_DETECTED")
            # elif results['gaze_direction'] != "Center":
            #     violation_type = "GAZE_AWAY"
            #     alert_system.speak_alert(violation_type)
                
            #     # Capture and log violation
            #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            #     violation_image = violation_capturer.capture_violation(frame, violation_type, timestamp)
            #     violation_logger.log_violation(
            #         violation_type,
            #         timestamp,
            #         {'duration': '5+ seconds', 'frame': results}
            #     )
                # alert_system.speak_alert("GAZE_AWAY")
            elif results['mouth_moving']:
                violation_type = "MOUTH_MOVING"
                alert_system.speak_alert(violation_type)
                
                # Capture and log violation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                violation_image = violation_capturer.capture_violation(frame, violation_type, timestamp)
                violation_logger.log_violation(
                    violation_type,
                    timestamp,
                    {'duration': '5+ seconds', 'frame': results}
                )
                # alert_system.speak_alert("MOUTH_MOVING")

            
            # Display and record
            display_detection_results(frame, results)
            video_recorder.record_frame(frame)
            
            # Show preview
            cv2.imshow('Exam Proctoring', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        violations = violation_logger.get_violations()
        report_path = report_generator.generate_report(student_info, violations)
        if report_path:
            print(f"\n{'='*60}")
            print(f"REPORT GENERATED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Report saved to: {os.path.abspath(report_path)}")
            print(f"Open this file in your web browser to view the report.")
            print(f"{'='*60}\n")
        else:
            print("Report generation failed")
            
        if config['screen']['recording']:
            screen_data = screen_recorder.stop_recording()
            if screen_data:
                print(f"Screen recording saved: {screen_data['filename']}")
        
        video_data = video_recorder.stop_recording()
        if video_data:
            print(f"Webcam recording saved: {video_data['filename']}")
        else:
            print("Video recording was not started or failed")
        
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"FATAL ERROR: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}")
        import sys
        sys.exit(1)
