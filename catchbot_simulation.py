"""CATCHBot end-to-end simulation.

Run from the repository root:
    python simulation/catchbot_simulation.py

Controls:
    SPACE - start anew randomized throw
    R     - reset the current throw
    Q/ESC - quit

The camera is fixed on a tripod. A randomized throw is generated, noisy
camera detections are collected, a parabolic trajectory is estimated, and the
virtual trash can moves toward the predicted landing location.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import cv2
import numpy as np

WIDTH, HEIGHT = 1100, 700
GROUND_Y = 560
CAN_WIDTH, CAN_HEIGHT = 105, 100
CAN_MAX_SPEED = 430.0
DETECTION_NOISE = 3.0


@dataclass
class Throw:
    x0: float
    y0: float
    vx: float
    vy: float
    gravity: float
    radius: int = 15

    def position(self, elapsed):
        return (self.x0 + self.vx * elapsed,
                self.y0 + self.vy * elapsed + 0.5 * self.gravity * elapsed ** 2)

    def landing_time(self, landing_y):
        a, b, c = 0.5 * self.gravity, self.vy, self.y0 - landing_y
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        roots = ((-b - math.sqrt(discriminant)) / (2 * a),
                 (-b + math.sqrt(discriminant)) / (2 * a))
        falling = [t for t in roots if t > 0 and self.vy + self.gravity * t > 0]
        return min(falling) if falling else None


class CatchBotSimulation:
    def __init__(self):
        self.rng = random.Random()
        self.can_x = WIDTH / 2
        self.throw = None
        self.last_frame = time.perf_counter()
        self.detections = []
        self.predicted_x = None
        self.actual_x = None
        self.status = "READY"
        self.frames_seen = 0
        self.new_throw()

    def new_throw(self):
        """Generate a varied throw to simulate different real-world inputs."""
        self.throw = Throw(
            x0=self.rng.uniform(180, 410),
            y0=self.rng.uniform(210, 330),
            vx=self.rng.uniform(170, 410),
            vy=self.rng.uniform(-840, -650),
            gravity=self.rng.uniform(920, 1120),
        )
        self.throw_started = time.perf_counter()
        landing_time = self.throw.landing_time(GROUND_Y)
        self.actual_x = self.throw.x0 + self.throw.vx * landing_time
        self.detections.clear()
        self.predicted_x = None
        self.status = "THROWING"
        self.frames_seen = 0

    def reset(self):
        self.can_x = WIDTH / 2
        self.new_throw()

    def detect_object(self, true_x, true_y):
        """Simulate a camera detector with small measurement noise."""
        return (true_x + self.rng.gauss(0, DETECTION_NOISE),
                true_y + self.rng.gauss(0, DETECTION_NOISE),
                self.throw.radius + self.rng.gauss(0, 1.0))

    def estimate_landing(self):
        """Fit x(t) and y(t), then calculate where the object crosses the catch line."""
        if len(self.detections) < 6:
            return
        recent = self.detections[-24:]
        t = np.array([row[0] for row in recent])
        t -= t[0]
        xs = np.array([row[1] for row in recent])
        ys = np.array([row[2] for row in recent])

        x_fit = np.polyfit(t, xs, 1)
        y_fit = np.polyfit(t, ys, 2)
        x0, vx = x_fit[1], x_fit[0]
        y0, vy, gravity = y_fit[2], y_fit[1], 2 * y_fit[0]
        if gravity <= 0:
            return

        a, b, c = 0.5 * gravity, vy, y0 - GROUND_Y
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return
        roots = ((-b - math.sqrt(discriminant)) / (2 * a),
                 (-b + math.sqrt(discriminant)) / (2 * a))
        falling = [root for root in roots if root > 0 and vy + gravity * root > 0]
        if falling:
            self.predicted_x = float(x0 + vx * min(falling))

    def move_can(self, target_x, dt):
        if target_x is None:
            return
        target_x = max(CAN_WIDTH / 2, min(WIDTH - CAN_WIDTH / 2, target_x))
        step = CAN_MAX_SPEED * dt
        difference = target_x - self.can_x
        self.can_x += max(-step, min(step, difference))

    def update(self, now):
        dt = min(now - self.last_frame, 0.05)
        self.last_frame = now
        if self.status in ("CAUGHT", "MISSED"):
            return

        elapsed = now - self.throw_started
        true_x, true_y = self.throw.position(elapsed)
        if true_y < GROUND_Y + self.throw.radius:
            detected_x, detected_y, _ = self.detect_object(true_x, true_y)
            self.detections.append((elapsed, detected_x, detected_y))
            self.frames_seen += 1
            self.estimate_landing()
            self.move_can(self.predicted_x, dt)
        else:
            self.status = "CAUGHT" if abs(self.can_x - true_x) <= CAN_WIDTH * 0.55 else "MISSED"

    def draw_camera_setup(self, frame):
        """Draw the fixed laptop camera mounted on a tripod."""
        center = WIDTH // 2
        cv2.line(frame, (center, HEIGHT - 1), (center - 105, HEIGHT), (70, 70, 70), 5)
        cv2.line(frame, (center, HEIGHT - 1), (center + 105, HEIGHT), (70, 70, 70), 5)
        cv2.line(frame, (center, HEIGHT - 1), (center, HEIGHT - 145), (70, 70, 70), 5)
        cv2.rectangle(frame, (center - 38, HEIGHT - 190, center + 38, HEIGHT - 145),
                      (30, 30, 30), -1)
        cv2.circle(frame, (center, HEIGHT - 168), 11, (100, 190, 230), -1)
        cv2.putText(frame, "FIXED TRIPOD CAMERA", (center - 118, HEIGHT - 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 2)

    def draw_trash_can(self, frame):
        left = int(self.can_x - CAN_WIDTH / 2)
        right = int(self.can_x + CAN_WIDTH / 2)
        top = GROUND_Y - CAN_HEIGHT
        cv2.ellipse(frame, (int(self.can_x), top), (CAN_WIDTH // 2, 14), (85, 105, 115), -1)
        cv2.rectangle(frame, (left, top), (right, GROUND_Y), (65, 85, 95), -1)
        cv2.ellipse(frame, (int(self.can_x), GROUND_Y), (CAN_WIDTH // 2, 15), (38, 52, 60), -1)
        cv2.rectangle(frame, (left - 7, top - 8), (right + 7, top + 8), (40, 52, 58), -1)
        cv2.putText(frame, "CATCHBot", (left + 10, top + 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (210, 230, 235), 2)

    def render(self):
        frame = np.full((HEIGHT, WIDTH, 3), (235, 244, 247), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (WIDTH, 92), (25, 49, 73), -1)
        cv2.putText(frame, "CATCHBot THROW-AND-CATCH SIMULATION", (24, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.88, (255, 255, 255), 2)
        cv2.putText(frame, "Randomized throw | Fixed tripod camera | No hardware required",
                    (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 228, 239), 1)
        cv2.line(frame, (0, GROUND_Y), (WIDTH, GROUND_Y), (130, 145, 150), 2)
        cv2.putText(frame, "CATCH LINE", (20, GROUND_Y - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (90, 105, 110), 1)

        elapsed = time.perf_counter() - self.throw_started
        true_x, true_y = self.throw.position(elapsed)
        if self.status == "THROWING":
            cv2.circle(frame, (int(true_x), int(true_y)), self.throw.radius, (40, 80, 220), -1)
            cv2.circle(frame, (int(true_x), int(true_y)), self.throw.radius + 4, (20, 30, 100), 2)
            if len(self.detections) > 1:
                points = np.array([(int(x), int(y)) for _, x, y in self.detections], dtype=np.int32)
                cv2.polylines(frame, [points], False, (180, 110, 220), 2)
            if self.predicted_x is not None:
                cv2.line(frame, (int(self.predicted_x), GROUND_Y - 35),
                         (int(self.predicted_x), GROUND_Y + 25), (0, 150, 255), 3)
                cv2.putText(frame, "PREDICTED LANDING", (int(self.predicted_x) - 82, GROUND_Y + 47),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 120, 210), 1)

        self.draw_trash_can(frame)
        self.draw_camera_setup(frame)
        predicted = "--" if self.predicted_x is None else f"{self.predicted_x:.0f}px"
        actual = "--" if self.actual_x is None else f"{self.actual_x:.0f}px"
        info = [f"STATUS: {self.status}", f"DETECTIONS: {self.frames_seen}",
                f"PREDICTED LANDING: {predicted}", f"ACTUAL LANDING: {actual}",
                f"CAN X: {self.can_x:.0f}px"]
        for index, text in enumerate(info):
            color = (50, 170, 80) if self.status == "CAUGHT" else (40, 50, 60)
            cv2.putText(frame, text, (22, 125 + index * 27), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2 if index == 0 else 1)
        cv2.putText(frame, "SPACE: new throw   R: reset   Q/ESC: quit", (WIDTH - 370, HEIGHT - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 65, 70), 1)
        return frame

    def run(self):
        while True:
            now = time.perf_counter()
            self.update(now)
            cv2.imshow("CATCHBot Simulation", self.render())
            key = cv2.waitKey(16) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                self.reset()
            elif key == ord(" "):
                self.new_throw()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    CatchBotSimulation().run()
