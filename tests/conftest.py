"""Avoid thread oversubscription on shared CPU test runners."""
import cv2

cv2.setNumThreads(1)
