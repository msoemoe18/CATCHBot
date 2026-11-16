import time

import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo11n.pt"  # smallest, fastest YOLO model; downloads automatically the first time
TARGET_CLASSES = {"sports ball", "bottle", "cup"}
MIN_CONFIDENCE = 0.35


class YoloDetector:
    """Deep-learning detector with the same output as detect(): (x, y, radius) or None."""

    def __init__(self, model_path=MODEL_PATH, targets=TARGET_CLASSES, conf=MIN_CONFIDENCE):
        self.model = YOLO(model_path)
        self.class_ids = [i for i, name in self.model.names.items() if name in targets]
        self.conf = conf
        self.last_label = None
        self.last_confidence = 0.0

    def detect(self, frame):
        result = self.model(frame, classes=self.class_ids, conf=self.conf, verbose=False)[0]
        if len(result.boxes) == 0:
            return None
        best = result.boxes[int(result.boxes.conf.argmax())]
        x1, y1, x2, y2 = best.xyxy[0].tolist()
        self.last_label = self.model.names[int(best.cls)]
        self.last_confidence = float(best.conf)
        radius = max(x2 - x1, y2 - y1) / 2
        return int((x1 + x2) / 2), int((y1 + y2) / 2), int(radius)


if __name__ == "__main__":
    detector = YoloDetector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera")

    fps = 0.0
    last_t = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t - last_t, 1e-6))
        last_t = t

        detection = detector.detect(frame)
        if detection is not None:
            x, y, r = detection
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.putText(frame, f"{detector.last_label} {detector.last_confidence:.2f}",
                        (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("YOLO", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if cv2.getWindowProperty("YOLO", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()