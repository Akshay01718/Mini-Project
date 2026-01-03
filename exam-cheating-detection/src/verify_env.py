import sys
print(f"Python: {sys.version}")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"OpenCV FAILED: {e}")

try:
    import mediapipe as mp
    print(f"MediaPipe: {mp.__version__}")
    try:
        from mediapipe import solutions
        print("from mediapipe import solutions: OK")
    except ImportError:
        print("from mediapipe import solutions: FAILED")
        
    try:
        import mediapipe.python.solutions
        print("import mediapipe.python.solutions: OK")
    except ImportError:
        print("import mediapipe.python.solutions: FAILED")
        
except ImportError as e:
    print(f"MediaPipe FAILED: {e}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"PyTorch FAILED: {e}")
