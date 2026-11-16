import json
from pathlib import Path

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
MIN_RADIUS = 5  # pixels; smaller blobs are treated as noise
KERNEL = np.ones((5, 5), np.uint8)


def load_hsv_range(path=CONFIG_PATH):
    with open(path) as f:
        config = json.load(f)
    return np.array(config["lower"]), np.array(config["upper"])


def detect(frame, lower, upper):
    """Find the largest blob in the color range. Returns (x, y, radius) or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest)
    if radius < MIN_RADIUS:
        return None
    return int(x), int(y), int(radius)


if __name__ == "__main__":
    lower, upper = load_hsv_range()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detection = detect(frame, lower, upper)
        if detection is not None:
            x, y, r = detection
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(frame, f"x={x} y={y} r={r}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No object", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()