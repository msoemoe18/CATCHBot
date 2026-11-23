
import math
import random
import time
from dataclasses import dataclass

import cv2
import numpy as np

from simulation_receiver import SimulationReceiver


# ---------------------------------------
# Window settings
# ---------------------------------------

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 760
HEADER_BOTTOM = 190
FLOOR_Y = 620

DISPLAY_WIDTH = 930
DISPLAY_HEIGHT = 589
WINDOW_X = 565
WINDOW_Y = 35

CAMERA_WIDTH = 700
CAMERA_HEIGHT = 350


# ---------------------------------------
# Trash can settings
# ---------------------------------------

CAN_WIDTH = 112
CAN_HEIGHT = 105
CAN_SPEED = 650.0
CAN_ACCELERATION = 1900.0
TARGET_SMOOTHING = 0.10

# ---------------------------------------
# Live tracking settings
# ---------------------------------------

TRACK_HOLD_SECONDS = 2.5
TRACK_MAX_POINTS = 600
TRACK_GAP_SECONDS = 1.5
DETECTION_FRESH_SECONDS = 0.45
PREDICTION_FRESH_SECONDS = 0.5

DETECTION_NOISE = 2.5


@dataclass
class Throw:
    x0: float
    y0: float
    vx: float
    vy: float
    gravity: float
    size: int = 17

    def position(self, elapsed):
        x = self.x0 + self.vx * elapsed
        y = (
            self.y0
            + self.vy * elapsed
            + 0.5 * self.gravity * elapsed ** 2
        )
        return x, y

    def landing_time(self, target_y):
        a = 0.5 * self.gravity
        b = self.vy
        c = self.y0 - target_y

        disc = b * b - 4 * a * c

        if disc < 0:
            return None

        roots = [
            (-b - math.sqrt(disc)) / (2 * a),
            (-b + math.sqrt(disc)) / (2 * a),
        ]

        valid = [
            t for t in roots
            if t > 0 and self.vy + self.gravity * t > 0
        ]

        return min(valid) if valid else None


class CatchBotSimulation:

    def __init__(self):
        self.random = random.Random()

        # Live tracking
        self.live_positions = []
        self.last_processed_track = 0.0
        self.live_prediction_x = None
        self.live_prediction_at = 0.0

        # UDP receiver
        self.receiver = SimulationReceiver()
        self.receiver.start()

        # Trash can
        self.can_x = 700.0
        self.can_velocity = 0.0
        self.smoothed_target = None

        # Simulation state
        self.mode = "LIVE"
        self.throw = None
        self.throw_started = time.perf_counter()
        self.last_frame = time.perf_counter()

        self.detections = []
        self.predicted_x = None
        self.actual_x = None
        self.status = "READY"
        self.frames_seen = 0

        if self.mode == "SIMULATION":
            self.new_throw()

    # ---------------------------------------
    # Simulation throw
    # ---------------------------------------

    def new_throw(self):
        if self.random.choice(("left", "right")) == "left":
            x0 = self.random.uniform(220, 350)
            vx = self.random.uniform(230, 410)
        else:
            x0 = self.random.uniform(850, 980)
            vx = -self.random.uniform(230, 410)

        self.throw = Throw(
            x0=x0,
            y0=self.random.uniform(330, 390),
            vx=vx,
            vy=self.random.uniform(-500, -390),
            gravity=self.random.uniform(980, 1160),
        )

        self.throw_started = time.perf_counter()
        self.detections.clear()
        self.predicted_x = None
        self.status = "THROWING"
        self.frames_seen = 0

        t = self.throw.landing_time(FLOOR_Y - 14)
        self.actual_x = (
            self.throw.x0 + self.throw.vx * t
            if t is not None else None
        )

        self.can_velocity = 0.0
        self.smoothed_target = None

    def reset(self):
        self.can_x = 700.0
        self.can_velocity = 0.0
        self.smoothed_target = None

        if self.mode == "SIMULATION":
            self.new_throw()
        else:
            self.live_positions.clear()
            self.last_processed_track = 0.0
            self.live_prediction_x = None
            self.status = "READY"

    def computer_vision_detection(self, x, y):
        return (
            x + self.random.gauss(0, DETECTION_NOISE),
            y + self.random.gauss(0, DETECTION_NOISE),
        )

    def estimate_landing(self):
        if len(self.detections) < 7:
            return

        recent = self.detections[-28:]
        t = np.array([p[0] for p in recent])
        t -= t[0]

        xs = np.array([p[1] for p in recent])
        ys = np.array([p[2] for p in recent])

        if np.ptp(t) < 0.001:
            return

        vx, x0 = np.polyfit(t, xs, 1)
        a, vy, y0 = np.polyfit(t, ys, 2)

        if a <= 0:
            return

        disc = vy ** 2 - 4 * a * (y0 - FLOOR_Y + 14)

        if disc < 0:
            return

        roots = [
            (-vy - math.sqrt(disc)) / (2 * a),
            (-vy + math.sqrt(disc)) / (2 * a),
        ]

        valid = [
            r for r in roots
            if r > 0 and vy + 2 * a * r > 0
        ]

        if valid:
            self.predicted_x = x0 + vx * min(valid)

    # ---------------------------------------
    # Smooth trash can movement
    # ---------------------------------------

    def move_trash_can(self, target_x, dt):
        if target_x is None or dt <= 0:
            return

        target_x = float(target_x)

        if not math.isfinite(target_x):
            return

        minimum = CAN_WIDTH / 2
        maximum = WINDOW_WIDTH - CAN_WIDTH / 2

        target_x = max(minimum, min(maximum, target_x))

        # Smooth changing predictions
        if self.smoothed_target is None:
            self.smoothed_target = target_x
        else:
            alpha = 1 - math.exp(
                -dt / TARGET_SMOOTHING
            )

            self.smoothed_target += alpha * (
                target_x - self.smoothed_target
            )

        difference = self.smoothed_target - self.can_x

        # Desired speed includes braking distance
        if abs(difference) > 0.5:
            desired_speed = math.copysign(
                min(
                    CAN_SPEED,
                    math.sqrt(
                        2 * CAN_ACCELERATION
                        * abs(difference)
                    ),
                ),
                difference,
            )
        else:
            desired_speed = 0.0

        max_change = CAN_ACCELERATION * dt

        change = desired_speed - self.can_velocity
        change = max(-max_change, min(max_change, change))

        self.can_velocity += change

        step = self.can_velocity * dt

        if (
            step * difference >= 0
            and abs(step) >= abs(difference)
        ):
            self.can_x = self.smoothed_target
            self.can_velocity = 0.0
        else:
            self.can_x += step

        self.can_x = max(
            minimum,
            min(maximum, self.can_x)
        )

    def brake_trash_can(self, dt):
        if abs(self.can_velocity) < 0.1:
            self.can_velocity = 0.0
            return

        braking = CAN_ACCELERATION * dt

        if abs(self.can_velocity) <= braking:
            self.can_velocity = 0.0
        else:
            self.can_velocity -= math.copysign(
                braking, self.can_velocity
            )

        self.can_x += self.can_velocity * dt
        self.can_x = max(
            CAN_WIDTH / 2,
            min(WINDOW_WIDTH - CAN_WIDTH / 2, self.can_x)
        )

    # ---------------------------------------
    # Live camera tracking
    # ---------------------------------------

    def update_live_tracking(self):
        position = self.receiver.latest_position
        received_at = self.receiver.last_track_received
        now = time.monotonic()

        # Keep the completed trail briefly
        if (
            not received_at
            or now - received_at > TRACK_HOLD_SECONDS
        ):
            self.live_positions.clear()
            self.last_processed_track = 0.0
            return

        if position is None:
            return

        # Do not add the same UDP message twice
        if received_at <= self.last_processed_track:
            return

        # New tracking sequence after a long gap
        if (
            self.last_processed_track > 0
            and received_at - self.last_processed_track
            > TRACK_GAP_SECONDS
        ):
            self.live_positions.clear()

        self.last_processed_track = received_at

        x = position[0] * WINDOW_WIDTH / CAMERA_WIDTH

        y = HEADER_BOTTOM + position[1] * (
            FLOOR_Y - HEADER_BOTTOM
        ) / CAMERA_HEIGHT

        if not (math.isfinite(x) and math.isfinite(y)):
            return

        point = (
            int(max(0, min(WINDOW_WIDTH - 1, x))),
            int(max(HEADER_BOTTOM, min(FLOOR_Y, y))),
        )

        if (
            not self.live_positions
            or self.live_positions[-1] != point
        ):
            self.live_positions.append(point)

        self.live_positions = self.live_positions[
            -TRACK_MAX_POINTS:
        ]

    # ---------------------------------------
    # Main update
    # ---------------------------------------

    def update(self, now):
        dt = max(0.001, min(now - self.last_frame, 0.05))
        self.last_frame = now

        if self.mode == "LIVE":
            self.update_live_tracking()

            external_x = self.receiver.latest_target

            fresh = (
                external_x is not None
                and time.monotonic()
                - self.receiver.last_received
                <= PREDICTION_FRESH_SECONDS
            )

            if fresh:
                try:
                    target = (
                        float(external_x)
                        * WINDOW_WIDTH / CAMERA_WIDTH
                    )

                    if math.isfinite(target):
                        self.live_prediction_x = target
                        self.live_prediction_at = time.monotonic()
                        self.move_trash_can(target, dt)
                    else:
                        self.brake_trash_can(dt)

                except (TypeError, ValueError):
                    self.brake_trash_can(dt)
            else:
                self.smoothed_target = None
                self.brake_trash_can(dt)

            return

        if self.status in ("CAUGHT", "MISSED"):
            return

        elapsed = now - self.throw_started
        true_x, true_y = self.throw.position(elapsed)

        if true_y < FLOOR_Y - 4:
            x, y = self.computer_vision_detection(
                true_x, true_y
            )

            self.detections.append((elapsed, x, y))
            self.frames_seen += 1

            self.estimate_landing()
            self.move_trash_can(self.predicted_x, dt)

        else:
            if abs(self.can_x - true_x) <= CAN_WIDTH * 0.58:
                self.status = "CAUGHT"
            else:
                self.status = "MISSED"

    # ---------------------------------------
    # Room drawing
    # ---------------------------------------

    def draw_room(self, frame):
        cv2.rectangle(
            frame,
            (0, HEADER_BOTTOM),
            (WINDOW_WIDTH, 475),
            (211, 222, 226),
            -1,
        )

        cv2.line(
            frame, (0, 475), (WINDOW_WIDTH, 475),
            (125, 137, 140), 5
        )

        cv2.rectangle(
            frame,
            (0, 510),
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            (163, 128, 96),
            -1,
        )

        for x in range(-200, WINDOW_WIDTH + 300, 75):
            cv2.line(
                frame,
                (x, 510),
                (x + 300, WINDOW_HEIGHT),
                (184, 148, 112),
                2,
            )

        # Window
        cv2.rectangle(
            frame, (70, 235), (300, 405),
            (82, 105, 120), -1
        )
        cv2.rectangle(
            frame, (82, 247), (288, 393),
            (150, 206, 225), -1
        )
        cv2.line(
            frame, (185, 247), (185, 393),
            (235, 240, 240), 5
        )
        cv2.line(
            frame, (82, 320), (288, 320),
            (235, 240, 240), 5
        )

        cv2.putText(
            frame, "ROOM CAMERA VIEW", (78, 430),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (70, 80, 84), 1
        )

        # Rug
        cv2.ellipse(
            frame, (600, 585), (360, 70),
            0, 0, 360, (121, 95, 74), -1
        )
        cv2.ellipse(
            frame, (600, 585), (350, 60),
            0, 0, 360, (139, 108, 82), 3
        )

    def draw_tripod_camera(self, frame):
        x, y = 825, 285

        for foot_x in (705, 825, 925):
            cv2.line(
                frame, (x, y + 40),
                (foot_x, 500), (48, 48, 48), 5
            )

        cv2.rectangle(
            frame,
            (x - 52, y - 28),
            (x + 52, y + 35),
            (32, 37, 40), -1
        )

        cv2.circle(
            frame, (x - 12, y + 3),
            17, (80, 175, 220), -1
        )

        cv2.circle(
            frame, (x - 12, y + 3),
            8, (15, 35, 50), -1
        )

        cv2.putText(
            frame, "FIXED TRIPOD CAMERA", (705, 255),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (45, 55, 60), 2
        )

    def draw_trash_can(self, frame):
        left = int(self.can_x - CAN_WIDTH / 2)
        right = int(self.can_x + CAN_WIDTH / 2)
        top = FLOOR_Y - CAN_HEIGHT

        cv2.ellipse(
            frame, (int(self.can_x), top),
            (CAN_WIDTH // 2, 15),
            0, 0, 360, (84, 105, 115), -1
        )

        cv2.rectangle(
            frame, (left, top), (right, FLOOR_Y),
            (63, 84, 94), -1
        )

        cv2.ellipse(
            frame, (int(self.can_x), FLOOR_Y),
            (CAN_WIDTH // 2, 16),
            0, 0, 360, (35, 49, 58), -1
        )

        cv2.rectangle(
            frame, (left - 8, top - 8),
            (right + 8, top + 8),
            (35, 48, 54), -1
        )

        cv2.putText(
            frame, "CATCHBot", (left + 11, top + 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (215, 230, 235), 2
        )

        if self.mode == "SIMULATION" and self.status == "CAUGHT":
            cv2.circle(
                frame, (int(self.can_x), top + 25),
                12, (30, 220, 80), -1
            )

    # ---------------------------------------
    # Tracking visualization
    # ---------------------------------------
    
    
    def draw_live_tracking(self, frame):
        """Draw the purple trail only where the bottle was detected."""

        if not self.live_positions:
            return

        # Actual positions received from the camera
        points = np.array(
            self.live_positions,
            dtype=np.int32
        )

        # Purple trail follows the bottle, never goes ahead
        if len(points) > 1:
            cv2.polylines(
                frame,
                [points],
                False,
                (255, 80, 210),
                3,
                cv2.LINE_AA
            )

        # Current detected bottle position
        x, y = self.live_positions[-1]

        # Only show the green box for a recent detection
        age = (
            time.monotonic()
            - self.receiver.last_track_received
        )

        if age <= 0.45:
            cv2.rectangle(
                frame,
                (x - 20, y - 20),
                (x + 20, y + 20),
                (40, 230, 70),
                3
            )

            cv2.putText(
                frame,
                "LIVE CAMERA OBJECT",
                (
                    max(5, x - 80),
                    max(HEADER_BOTTOM + 20, y - 30)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (40, 230, 70),
                2
            )


    def draw_detection_overlay(self, frame):
        if not self.detections:
            return

        points = np.array([
            (int(p[1]), int(p[2]))
            for p in self.detections
            if HEADER_BOTTOM <= p[2] <= FLOOR_Y
        ], dtype=np.int32)

        if len(points) > 1:
            cv2.polylines(
                frame, [points], False,
                (255, 80, 210), 3
            )

        if len(points) > 0:
            x, y = points[-1]
            cv2.rectangle(
                frame,
                (x - 17, y - 17),
                (x + 17, y + 17),
                (40, 230, 70), 3
            )

    # ---------------------------------------
    # Header and status panels
    # ---------------------------------------

    def draw_header(self, frame):
        cv2.rectangle(
            frame, (0, 0),
            (WINDOW_WIDTH, HEADER_BOTTOM),
            (180, 90, 30), -1
        )

        cv2.putText(
            frame, "CATCHBot", (25, 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.92,
            (255, 255, 255), 2
        )

        cv2.putText(
            frame,
            "COMPUTER-VISION THROW SIMULATION",
            (190, 37),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (225, 248, 255), 1
        )

        cv2.putText(
            frame,
            "Fixed camera | Detect | Track | Predict | Catch",
            (27, 69),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (220, 246, 255), 1
        )

        # Tracking panel
        cv2.rectangle(
            frame, (20, 83), (510, 174),
            (42, 93, 125), -1
        )
        cv2.rectangle(
            frame, (20, 83), (510, 174),
            (190, 235, 250), 2
        )

        if self.mode == "LIVE":
            age = (
                time.monotonic()
                - self.receiver.last_track_received
            )

            tracking = bool(
                self.live_positions
                and age <= DETECTION_FRESH_SECONDS
            )

            status = "TRACKING" if tracking else "READY"
            frames = len(self.live_positions)

            if (
                self.live_prediction_x is not None
                and time.monotonic() - self.live_prediction_at
                <= PREDICTION_FRESH_SECONDS
            ):
                predicted = f"{self.live_prediction_x:.0f}px"
            else:
                predicted = "--"

            actual = "--"
        else:
            status = self.status
            frames = self.frames_seen
            predicted = (
                "--" if self.predicted_x is None
                else f"{self.predicted_x:.0f}px"
            )
            actual = (
                "--" if self.actual_x is None
                else f"{self.actual_x:.0f}px"
            )

        cv2.putText(
            frame, "LIVE TRACKING", (38, 108),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (255, 255, 255), 2
        )

        cv2.putText(
            frame, f"STATUS: {status}", (38, 134),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 2
        )

        cv2.putText(
            frame, f"Frames: {frames}", (190, 134),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (235, 250, 255), 1
        )

        cv2.putText(
            frame, f"Predicted X: {predicted}", (300, 134),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (235, 250, 255), 1
        )

        cv2.putText(
            frame,
            f"Actual X: {actual}   Can X: {self.can_x:.0f}px",
            (38, 158),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43,
            (220, 242, 250), 1
        )

        # Controls panel
        cv2.rectangle(
            frame, (530, 83), (1180, 174),
            (42, 93, 125), -1
        )
        cv2.rectangle(
            frame, (530, 83), (1180, 174),
            (190, 235, 250), 2
        )

        cv2.putText(
            frame, "SIMULATION CONTROLS", (550, 108),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (255, 255, 255), 2
        )

        cv2.putText(
            frame,
            f"MODE: {self.mode}",
            (550, 136),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (255, 255, 255), 2
        )

        cv2.putText(
            frame,
            "M: switch mode   SPACE: throw   R: reset   Q/ESC: quit",
            (550, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44,
            (225, 245, 252), 1
        )

    # ---------------------------------------
    # Rendering
    # ---------------------------------------

    def render(self):
        frame = np.zeros(
            (WINDOW_HEIGHT, WINDOW_WIDTH, 3),
            dtype=np.uint8
        )

        for y in range(WINDOW_HEIGHT):
            color = int(235 - 35 * y / WINDOW_HEIGHT)
            frame[y, :] = (color, 244, 250)

        self.draw_room(frame)
        self.draw_header(frame)

        if self.mode == "LIVE":
            self.draw_live_tracking(frame)
        else:
            self.draw_detection_overlay(frame)

            if self.throw is not None:
                elapsed = time.perf_counter() - self.throw_started
                x, y = self.throw.position(elapsed)

                if (
                    self.status == "THROWING"
                    and HEADER_BOTTOM + self.throw.size
                    <= y <= FLOOR_Y
                ):
                    cv2.circle(
                        frame,
                        (int(x), int(y)),
                        self.throw.size,
                        (35, 55, 220), -1
                    )

            if (
                self.predicted_x is not None
                and self.status == "THROWING"
            ):
                px = int(self.predicted_x)

                cv2.line(
                    frame,
                    (px, FLOOR_Y - 55),
                    (px, FLOOR_Y + 15),
                    (0, 150, 255), 4
                )

            if self.status in ("CAUGHT", "MISSED"):
                message = (
                    "TRASH CAUGHT!"
                    if self.status == "CAUGHT"
                    else "THROW MISSED"
                )

                color = (
                    (30, 190, 70)
                    if self.status == "CAUGHT"
                    else (30, 50, 210)
                )

                cv2.putText(
                    frame, message, (450, 390),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, color, 3
                )

        self.draw_trash_can(frame)
        self.draw_tripod_camera(frame)

        # Bottom legend
        cv2.rectangle(
            frame,
            (0, WINDOW_HEIGHT - 47),
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            (23, 47, 70), -1
        )

        legend = [
            ("GREEN BOX = DETECTION", 25, (115, 255, 125)),
            ("PURPLE = TRACK", 260, (255, 130, 225)),
            ("ORANGE = PREDICTION", 430, (80, 180, 255)),
        ]

        for text, x, color in legend:
            cv2.putText(
                frame, text,
                (x, WINDOW_HEIGHT - 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46, color, 1
            )

        return frame

    # ---------------------------------------
    # Application
    # ---------------------------------------

    def run(self):
        window_name = "CATCHBot Simulation"

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_AUTOSIZE
        )

        cv2.moveWindow(
            window_name,
            WINDOW_X,
            WINDOW_Y
        )

        try:
            while True:
                now = time.perf_counter()
                self.update(now)

                display_frame = cv2.resize(
                    self.render(),
                    (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                    interpolation=cv2.INTER_AREA
                )

                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(16) & 0xFF

                if key in (ord("q"), 27):
                    break

                if key == ord("m"):
                    if self.mode == "LIVE":
                        self.mode = "SIMULATION"
                        self.reset()
                    else:
                        self.mode = "LIVE"
                        self.reset()

                elif key == ord("r"):
                    self.reset()

                elif key == ord(" ") and self.mode == "SIMULATION":
                    self.new_throw()

                if cv2.getWindowProperty(
                    window_name,
                    cv2.WND_PROP_VISIBLE
                ) < 1:
                    break

        finally:
            self.receiver.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    CatchBotSimulation().run()
