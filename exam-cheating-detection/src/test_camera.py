import cv2
import time

print("Testing Webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera (index 0)")
    # Try index 1 just in case
    print("Trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Could not open camera (index 1)")
        exit(1)

print("Camera backend opened successfully!")
print(f"Backend: {cap.getBackendName()}")

ret, frame = cap.read()
if ret:
    print(f"Successfully read frame! Shape: {frame.shape}")
    cv2.imshow("Camera Test", frame)
    print("Press any key to close window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("ERROR: Camera opened but failed to read frame.")

cap.release()
print("Done.")
