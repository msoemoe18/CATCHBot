import time

import cv2
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

from detector import detect, load_hsv_range

G = 9.81                 # gravity, m/s^2
BALL_RADIUS_M = 0.02     # 1.57 in foam ball -> 2 cm radius
MEAS_NOISE_PX = 3.0      # how many pixels the detector jitters
ACCEL_NOISE_PX = 2000.0  # unmodeled acceleration allowed (hand pushes, drag), px/s^2


class BallKalman:
    """Kalman filter in pixel coordinates. State: [x, y, vx, vy]."""

    def __init__(self, x, y):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([x, y, 0.0, 0.0])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]], dtype=float)
        self.kf.P = np.diag([MEAS_NOISE_PX**2, MEAS_NOISE_PX**2, 1e6, 1e6])
        self.kf.R = np.eye(2) * MEAS_NOISE_PX**2

    @staticmethod
    def gravity_px(radius_px):
        """Gravity in pixels/s^2 at the ball's current distance from the camera."""
        return G * radius_px / BALL_RADIUS_M

    def predict(self, dt, radius_px):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        B = np.array([0, 0.5 * dt**2, 0, dt])
        Q = Q_discrete_white_noise(dim=2, dt=dt, var=ACCEL_NOISE_PX**2,
                                   block_size=2, order_by_dim=False)
        self.kf.predict(u=self.gravity_px(radius_px), B=B, F=F, Q=Q)

    def update(self, x, y):
        self.kf.update(np.array([x, y], dtype=float))

    def future_path(self, radius_px, seconds=0.5, steps=15):
        """Where the ball would go over the next few seconds if it were in free flight."""
        x, y, vx, vy = self.kf.x
        g = self.gravity_px(radius_px)
        t = np.linspace(0, seconds, steps)
        return np.column_stack([x + vx * t, y + vy * t + 0.5 * g * t**2])


if __name__ == "__main__":
    lower, upper = load_hsv_range()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera")

    kf = None
    radius = 10
    missing = 0
    last_t = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter()
        dt = t - last_t
        last_t = t

        detection = detect(frame, lower, upper)

        if kf is not None:
            kf.predict(dt, radius)
        if detection is not None:
            x, y, radius = detection
            missing = 0
            if kf is None:
                kf = BallKalman(x, y)
            else:
                kf.update(x, y)
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
        else:
            missing += 1
            if missing > 5:
                kf = None

        if kf is not None:
            kx, ky, vx, vy = kf.kf.x
            path = kf.future_path(radius).astype(np.int32)
            cv2.polylines(frame, [path], False, (255, 0, 255), 2)
            cv2.circle(frame, (int(kx), int(ky)), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"vx={vx:.0f} vy={vy:.0f} px/s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Kalman", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()