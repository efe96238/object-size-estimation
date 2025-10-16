import cv2
import numpy as np
from reference_detection import detect_reference_object
from size_calculation import estimate_object_size

def main():
    cap = cv2.VideoCapture(0) 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Now correctly unpack three values
        reference_size, reference_contour, real_world_size = detect_reference_object(frame)

        if reference_size is not None:
            object_sizes = estimate_object_size(frame, reference_size, reference_contour, real_world_size)

            for obj in object_sizes:
                x, y, w, h, size_text = obj
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, size_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Object Size Estimation", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
