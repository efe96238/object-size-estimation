import cv2
import numpy as np
from reference_detection import detect_reference_object

def estimate_object_size(frame, reference_size, reference_contour, real_world_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_sizes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if (x, y, w, h) == reference_contour:
            continue  # Skip the reference object

        # Convert pixels to real-world size using the correct reference dimension
        size_cm_width = (w / reference_size) * real_world_size
        size_cm_height = (h / reference_size) * real_world_size
        size_text = f"{size_cm_width:.2f} cm x {size_cm_height:.2f} cm"
        
        object_sizes.append((x, y, w, h, size_text))

    return object_sizes
