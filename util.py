import cv2
import numpy as np

def get_color_limits(color):
    """
    Given a color in BGR, returns the lower and upper HSV bounds.
    """
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    lower_limit = np.array([hsvC[0][0][0] - 10, 100, 100], dtype=np.uint8)  # Adjusted for color detection
    upper_limit = np.array([hsvC[0][0][0] + 10, 255, 255], dtype=np.uint8)
    return lower_limit, upper_limit
