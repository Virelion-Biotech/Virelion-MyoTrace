"""Avoid thread oversubscription on shared CPU test runners."""
try:
    import cv2
except ImportError:  # OpenCV is optional ([video] extra); video tests skip themselves.
    cv2 = None

if cv2 is not None:
    cv2.setNumThreads(1)
