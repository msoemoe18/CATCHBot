import json
import cv2
import numpy as np

CONFIG_PATH = "config.json"
WINDOW = "Tuner"

def nothing(_):
    pass

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Could not open the camera")

cv2.namedWindow(WINDOW)
sliders = [
    ("H min", 35, 179), ("H max", 85, 179),
    ("S min", 80, 255), ("S max", 255, 255),
    ("V min", 60, 255), ("V max", 255, 255),
]
for name, start, maximum in sliders:
    cv2.createTrackbar(name, WINDOW, start, maximum, nothing)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pos = {name: cv2.getTrackbarPos(name, WINDOW) for name, _, _ in sliders}
    lower = np.array([pos["H min"], pos["S min"], pos["V min"]])
    upper = np.array([pos["H max"], pos["S max"], pos["V max"]])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        with open(CONFIG_PATH, "w") as f:
            json.dump({"lower": lower.tolist(), "upper": upper.tolist()}, f, indent=2)
        print("Saved", lower.tolist(), upper.tolist(), "to", CONFIG_PATH)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()