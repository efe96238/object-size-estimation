import cv2
import numpy as np

# in cm
KNOWN_WIDTH_CM = 8.56  # Long side
KNOWN_HEIGHT_CM = 5.4   # Short side
ASPECT_RATIO_RANGE = (1.5, 1.8)  
MIN_AREA = 3000  # Minimum contour area to be considered
MAX_AREA = 30000  # Maximum contour area to filter out huge objects

def detect_reference_object(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reference_size = None
    reference_contour = None
    real_world_size = None
    best_contour = None
    max_contour_area = 0  

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_AREA or area > MAX_AREA:
            continue  # Ignore too small or too large objects

        # Approximate contour to detect shape
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:  # Only accept rectangles
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            if ASPECT_RATIO_RANGE[0] <= aspect_ratio <= ASPECT_RATIO_RANGE[1]:
                if area > max_contour_area:  # Select the largest valid rectangle
                    max_contour_area = area
                    best_contour = (x, y, w, h)

    if best_contour:
        x, y, w, h = best_contour

        if w > h:  
            reference_size = w  # Horizontal position → use width
            real_world_size = KNOWN_WIDTH_CM
        else:
            reference_size = h  # Vertical position → use height
            real_world_size = KNOWN_HEIGHT_CM

        reference_contour = best_contour

    return reference_size, reference_contour, real_world_size
