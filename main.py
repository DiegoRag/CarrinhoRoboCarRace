import cv2
import numpy as np
import serial
import time
import os
from util import get_color_limits

# Initialize serial communication
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
ser.reset_input_buffer()

# Focal length and object width (in cm)
FOCAL_LENGTH = 1421
OBJECT_WIDTH = 10.1  # Height of the object in real-world units (cm)

# Define color limits for green and red cones
green = [156, 165, 137]  # Green in BGR
red = [113, 109, 180]  # Red in BGR

# Get HSV limits for the colors
green_lower, green_upper = get_color_limits(green)
red_lower, red_upper = get_color_limits(red)

# Initialize video capture
cap = cv2.VideoCapture(0)

def detect_and_calculate_distance(mask, color_name, frame):
    """Detects the object and calculates the distance."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate the distance using the bounding box height
        distance = (OBJECT_WIDTH * FOCAL_LENGTH) / h  # Distance in cm

        # Draw bounding box and distance label
        color = (0, 255, 0) if color_name == "Green" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{color_name} Dist: {h:.2f} centimeters",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return distance
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green detection
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    green_distance = detect_and_calculate_distance(green_mask, "Green", frame)

    # Red detection
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    red_distance = detect_and_calculate_distance(red_mask, "Red", frame)

    # Show the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
