import time
from collections import deque

import cv2
import numpy as np

from detector import detect, load_hsv_range


class Tracker:
    """Remembers recent detections and estimates velocity in pixels per second."""

    def __init__(self, max_points=30, max_missing=5):
        self.points = deque(maxlen=max_points)  # each item: (t, x, y, r)
        self.max_missing = max_missing
        self.missing = 0

    def update(self, detection, t):
        if detection is None:
            self.missing += 1
            if self.missing > self.max_missing:
                self.points.clear()
            return
        self.missing = 0
        x, y, r = detection
        self.points.append((t, x, y, r))

    def velocity(self, n=5):
        """Best-fit slope of x and y over the last n points. Returns (vx, vy) or None."""
        if len(self.points) < 3:
            return None
        recent = np.array(self.points)[-n:]
        t = recent[:, 0] - recent[0, 0]
        if t[-1] == 0:
            return None
        vx = np.polyfit(t, recent[:, 1], 1)[0]
        vy = np.polyfit(t, recent[:, 2], 1)[0]
        return vx, vy

    def trail(self):
        return [(int(x), int(y)) for _, x, y, _ in self.points]


if __name__ == "__main__":
    lower, upper = load_hsv_range()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera")

    tracker = Tracker()
    fps = 0.0
    last_t = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t - last_t, 1e-6))
        last_t = t

        detection = detect(frame, lower, upper)
        tracker.update(detection, t)

        trail = tracker.trail()
        if len(trail) >= 2:
            cv2.polylines(frame, [np.array(trail)], False, (255, 0, 0), 2)

        if detection is not None:
            x, y, r = detection
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            v = tracker.velocity()
            if v is not None:
                vx, vy = v
                tip = (int(x + vx * 0.1), int(y + vy * 0.1))
                cv2.arrowedLine(frame, (x, y), tip, (0, 0, 255), 2)
                cv2.putText(frame, f"vx={vx:.0f} vy={vy:.0f} px/s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()