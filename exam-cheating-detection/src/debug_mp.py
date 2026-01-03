import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"CWD: {os.getcwd()}")

try:
    import mediapipe
    print(f"MediaPipe imported successfully")
    print(f"File: {getattr(mediapipe, '__file__', 'Unknown')}")
    print(f"Path: {getattr(mediapipe, '__path__', 'Unknown')}")
    print(f"Dir: {dir(mediapipe)}")
    
    if hasattr(mediapipe, 'solutions'):
        print("mediapipe.solutions exists")
    else:
        print("mediapipe.solutions DOES NOT EXIST")
        
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
