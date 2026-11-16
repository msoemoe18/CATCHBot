import math
import random
import time
from dataclasses import dataclass

import cv2
import numpy as np


WIDTH, HEIGHT = 1200, 760
ROOM_TOP = 92
FLOOR_Y = 620
CAN_WIDTH = 112
CAN_HEIGHT = 105
CAN_SPEED = 430
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
        y = self.y0 + self.vy * elapsed
        y += 0.5 * self.gravity * elapsed**2
        return x, y

    def landing_time(self, target_y):
        a = 0.5 * self.gravity
        b = self.vy
        c = self.y0 - target_y

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return None

        roots = (
            (-b - math.sqrt(discriminant)) / (2 * a),
            (-b + math.sqrt(discriminant)) / (2 * a),
        )

        falling_roots = [
            root
            for root in roots
            if root > 0 and self.vy + self.gravity * root > 0
        ]

        return min(falling_roots) if falling_roots else None


class CatchBotSimulation:
    def __init__(self):
        self.random = random.Random()

        self.can_x = 700.0
        self.throw = None
        self.throw_started = time.perf_counter()
        self.last_frame = time.perf_counter()

        # time, center_x, center_y, width, height
        self.detections = []

        self.predicted_x = None
        self.actual_x = None
        self.status = "READY"
        self.frames_seen = 0

        self.new_throw()

    def new_throw(self):
        # Randomly throw the trash from the left or right side.
        if self.random.choice(["left", "right"]) == "left":
            start_x = self.random.uniform(190, 355)
            horizontal_velocity = self.random.uniform(230, 410)
        else:
            start_x = self.random.uniform(845, 1010)
            horizontal_velocity = -self.random.uniform(230, 410)

        self.throw = Throw(
            x0=start_x,
            y0=self.random.uniform(280, 390),
            vx=horizontal_velocity,
            vy=self.random.uniform(-920, -730),
            gravity=self.random.uniform(980, 1160),
        )

        self.throw_started = time.perf_counter()
        self.detections.clear()
        self.predicted_x = None
        self.status = "THROWING"
        self.frames_seen = 0

        landing_time = self.throw.landing_time(FLOOR_Y - 14)

        if landing_time is not None:
            self.actual_x = (
                self.throw.x0
                + self.throw.vx * landing_time
            )
        else:
            self.actual_x = None

    def reset(self):
        self.can_x = 700.0
        self.new_throw()

    def computer_vision_detection(self, true_x, true_y):
        # Simulated camera detection with small measurement noise.
        center_x = true_x + self.random.gauss(0, DETECTION_NOISE)
        center_y = true_y + self.random.gauss(0, DETECTION_NOISE)

        box_width = self.throw.size * 2 + self.random.randint(-2, 3)
        box_height = self.throw.size * 2 + self.random.randint(-2, 3)

        return center_x, center_y, box_width, box_height

    def estimate_landing(self):
        if len(self.detections) < 7:
            return

        recent = self.detections[-28:]

        times = np.array(
            [item[0] for item in recent],
            dtype=float
        )
        times -= times[0]

        xs = np.array(
            [item[1] for item in recent],
            dtype=float
        )

        ys = np.array(
            [item[2] for item in recent],
            dtype=float
        )

        x_fit = np.polyfit(times, xs, 1)
        y_fit = np.polyfit(times, ys, 2)

        x0 = x_fit[1]
        velocity_x = x_fit[0]

        y0 = y_fit[2]
        velocity_y = y_fit[1]
        gravity = 2 * y_fit[0]

        if gravity <= 0:
            return

        target_y = FLOOR_Y - 14

        a = 0.5 * gravity
        b = velocity_y
        c = y0 - target_y

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return

        roots = (
            (-b - math.sqrt(discriminant)) / (2 * a),
            (-b + math.sqrt(discriminant)) / (2 * a),
        )

        falling_roots = [
            root
            for root in roots
            if root > 0 and velocity_y + gravity * root > 0
        ]

        if falling_roots:
            landing_time = min(falling_roots)

            self.predicted_x = (
                x0 + velocity_x * landing_time
            )

    def move_trash_can(self, target_x, dt):
        if target_x is None:
            return

        minimum_x = CAN_WIDTH / 2
        maximum_x = WIDTH - CAN_WIDTH / 2

        target_x = max(
            minimum_x,
            min(maximum_x, target_x)
        )

        movement = CAN_SPEED * dt
        difference = target_x - self.can_x

        self.can_x += max(
            -movement,
            min(movement, difference)
        )

    def update(self, now):
        dt = min(
            now - self.last_frame,
            0.05
        )

        self.last_frame = now

        if self.status in ["CAUGHT", "MISSED"]:
            return

        elapsed = now - self.throw_started
        true_x, true_y = self.throw.position(elapsed)

        if true_y < FLOOR_Y - 4:
            detection = self.computer_vision_detection(
                true_x,
                true_y
            )

            center_x, center_y, box_width, box_height = detection

            self.detections.append(
                (
                    elapsed,
                    center_x,
                    center_y,
                    box_width,
                    box_height
                )
            )

            self.frames_seen += 1
            self.estimate_landing()
            self.move_trash_can(
                self.predicted_x,
                dt
            )

        else:
            catch_distance = CAN_WIDTH * 0.58

            if abs(self.can_x - true_x) <= catch_distance:
                self.status = "CAUGHT"
            else:
                self.status = "MISSED"

    def draw_room(self, frame):
        # White wall area
        cv2.rectangle(
            frame,
            (0, ROOM_TOP),
            (WIDTH, 475),
            (211, 222, 226),
            -1
        )

        # Baseboard
        cv2.line(
            frame,
            (0, 475),
            (WIDTH, 475),
            (125, 137, 140),
            5
        )

        cv2.line(
            frame,
            (0, 505),
            (WIDTH, 505),
            (185, 164, 143),
            12
        )

        # Floor
        cv2.rectangle(
            frame,
            (0, 510),
            (WIDTH, HEIGHT),
            (163, 128, 96),
            -1
        )

        for x in range(-200, WIDTH + 300, 75):
            cv2.line(
                frame,
                (x, 510),
                (x + 300, HEIGHT),
                (184, 148, 112),
                2
            )

        # Window
        cv2.rectangle(
            frame,
            (70, 145),
            (300, 315),
            (82, 105, 120),
            -1
        )

        cv2.rectangle(
            frame,
            (82, 157),
            (288, 303),
            (150, 206, 225),
            -1
        )

        cv2.line(
            frame,
            (185, 157),
            (185, 303),
            (235, 240, 240),
            5
        )

        cv2.line(
            frame,
            (82, 230),
            (288, 230),
            (235, 240, 240),
            5
        )

        cv2.putText(
            frame,
            "ROOM CAMERA VIEW",
            (78, 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (70, 80, 84),
            1
        )

        # Wall picture
        cv2.rectangle(
            frame,
            (450, 145),
            (635, 275),
            (88, 62, 45),
            -1
        )

        cv2.rectangle(
            frame,
            (465, 160),
            (620, 260),
            (227, 193, 131),
            -1
        )

        cv2.line(
            frame,
            (480, 240),
            (535, 190),
            (80, 130, 90),
            8
        )

        cv2.line(
            frame,
            (535, 190),
            (600, 235),
            (80, 130, 90),
            8
        )

        # Sofa
        cv2.rectangle(
            frame,
            (845, 350),
            (1125, 505),
            (73, 95, 110),
            -1
        )

        cv2.rectangle(
            frame,
            (870, 315),
            (1100, 400),
            (82, 107, 122),
            -1
        )

        cv2.rectangle(
            frame,
            (865, 480),
            (900, 540),
            (55, 55, 55),
            -1
        )

        cv2.rectangle(
            frame,
            (1070, 480),
            (1105, 540),
            (55, 55, 55),
            -1
        )

        # Rug
        cv2.ellipse(
            frame,
            (600, 585),
            (360, 70),
            0,
            0,
            360,
            (121, 95, 74),
            -1
        )

        cv2.ellipse(
            frame,
            (600, 585),
            (350, 60),
            0,
            0,
            360,
            (139, 108, 82),
            3
        )

    def draw_tripod_camera(self, frame):
        camera_x = 1010
        camera_y = 160

        cv2.line(
            frame,
            (camera_x, camera_y + 40),
            (camera_x - 120, 470),
            (48, 48, 48),
            5
        )

        cv2.line(
            frame,
            (camera_x, camera_y + 40),
            (camera_x + 100, 470),
            (48, 48, 48),
            5
        )

        cv2.line(
            frame,
            (camera_x, camera_y + 40),
            (camera_x, 470),
            (48, 48, 48),
            5
        )

        cv2.rectangle(
            frame,
            (camera_x - 52, camera_y - 28),
            (camera_x + 52, camera_y + 35),
            (32, 37, 40),
            -1
        )

        cv2.circle(
            frame,
            (camera_x - 12, camera_y + 3),
            17,
            (80, 175, 220),
            -1
        )

        cv2.circle(
            frame,
            (camera_x - 12, camera_y + 3),
            8,
            (15, 35, 50),
            -1
        )

        cv2.putText(
            frame,
            "FIXED COMPUTER-VISION CAMERA",
            (785, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (45, 55, 60),
            2
        )

    def draw_trash_can(self, frame):
        left = int(
            self.can_x - CAN_WIDTH / 2
        )

        right = int(
            self.can_x + CAN_WIDTH / 2
        )

        top = FLOOR_Y - CAN_HEIGHT

        # Lid
        cv2.ellipse(
            frame,
            (int(self.can_x), top),
            (CAN_WIDTH // 2, 15),
            0,
            0,
            360,
            (84, 105, 115),
            -1
        )

        # Body
        cv2.rectangle(
            frame,
            (left, top),
            (right, FLOOR_Y),
            (63, 84, 94),
            -1
        )

        # Bottom
        cv2.ellipse(
            frame,
            (int(self.can_x), FLOOR_Y),
            (CAN_WIDTH // 2, 16),
            0,
            0,
            360,
            (35, 49, 58),
            -1
        )

        # Handle
        cv2.rectangle(
            frame,
            (left - 8, top - 8),
            (right + 8, top + 8),
            (35, 48, 54),
            -1
        )

        cv2.putText(
            frame,
            "CATCHBot",
            (left + 11, top + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (215, 230, 235),
            2
        )

        if self.status == "CAUGHT":
            cv2.circle(
                frame,
                (int(self.can_x), top + 25),
                12,
                (30, 220, 80),
                -1
            )

    def draw_detection_overlay(self, frame):
        if not self.detections:
            return

        # Keep trajectory below the title bar.
        room_detections = [
            row
            for row in self.detections
            if ROOM_TOP <= row[2] <= FLOOR_Y
        ]

        if not room_detections:
            return

        # Purple trajectory trace
        trace = np.array(
            [
                (int(row[1]), int(row[2]))
                for row in room_detections
            ],
            dtype=np.int32
        )

        if len(trace) > 1:
            cv2.polylines(
                frame,
                [trace],
                False,
                (255, 80, 210),
                3
            )

        # Current bounding box
        _, center_x, center_y, box_width, box_height = (
            room_detections[-1]
        )

        x1 = int(center_x - box_width / 2)
        y1 = int(center_y - box_height / 2)
        x2 = int(center_x + box_width / 2)
        y2 = int(center_y + box_height / 2)

        # Green computer-vision bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (40, 230, 70),
            3
        )

        # Detected center point
        cv2.circle(
            frame,
            (int(center_x), int(center_y)),
            4,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            frame,
            "CV DETECTION: TRASH",
            (x1, y1 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 170, 45),
            2
        )

        cv2.putText(
            frame,
            f"center=({int(center_x)}, {int(center_y)})",
            (x1, y2 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 120, 35),
            1
        )

    def render(self):
        frame = np.full(
            (HEIGHT, WIDTH, 3),
            (235, 242, 245),
            dtype=np.uint8
        )

        # Title bar
        cv2.rectangle(
            frame,
            (0, 0),
            (WIDTH, ROOM_TOP),
            (23, 47, 70),
            -1
        )

        cv2.putText(
            frame,
            "CATCHBot COMPUTER-VISION ROOM SIMULATION",
            (25, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Fixed tripod camera | Bounding-box tracking | Trajectory prediction",
            (26, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (205, 228, 239),
            1
        )

        self.draw_room(frame)

        elapsed = time.perf_counter() - self.throw_started
        true_x, true_y = self.throw.position(elapsed)

        # Red thrown trash
        if self.status == "THROWING":
            cv2.circle(
                frame,
                (int(true_x), int(true_y)),
                self.throw.size,
                (35, 55, 220),
                -1
            )

            cv2.circle(
                frame,
                (int(true_x), int(true_y)),
                self.throw.size + 4,
                (20, 30, 100),
                2
            )

        self.draw_detection_overlay(frame)

        # Predicted landing line
        if (
            self.predicted_x is not None
            and self.status == "THROWING"
        ):
            cv2.line(
                frame,
                (int(self.predicted_x), FLOOR_Y - 55),
                (int(self.predicted_x), FLOOR_Y + 15),
                (0, 150, 255),
                4
            )

            cv2.putText(
                frame,
                "PREDICTED LANDING",
                (int(self.predicted_x) - 83, FLOOR_Y + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 115, 215),
                1
            )

        self.draw_trash_can(frame)
        self.draw_tripod_camera(frame)

        predicted = (
            "--"
            if self.predicted_x is None
            else f"{self.predicted_x:.0f}px"
        )

        actual = (
            "--"
            if self.actual_x is None
            else f"{self.actual_x:.0f}px"
        )

        status_color = (
            (40, 185, 70)
            if self.status == "CAUGHT"
            else (45, 55, 60)
        )

        panel = [
            f"STATUS: {self.status}",
            f"CV FRAMES TRACKED: {self.frames_seen}",
            f"PREDICTED LANDING: {predicted}",
            f"ACTUAL LANDING: {actual}",
            f"TRASH CAN X: {self.can_x:.0f}px",
        ]

        cv2.rectangle(
            frame,
            (18, 112),
            (350, 275),
            (241, 247, 248),
            -1
        )

        cv2.rectangle(
            frame,
            (18, 112),
            (350, 275),
            (100, 120, 125),
            2
        )

        for index, text in enumerate(panel):
            cv2.putText(
                frame,
                text,
                (33, 143 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.51,
                status_color if index == 0 else (45, 55, 60),
                2 if index == 0 else 1
            )

        if self.status == "CAUGHT":
            cv2.putText(
                frame,
                "TRASH CAUGHT!",
                (455, 365),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.15,
                (30, 190, 70),
                3
            )

        elif self.status == "MISSED":
            cv2.putText(
                frame,
                "THROW MISSED",
                (490, 365),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (30, 50, 210),
                3
            )

        cv2.putText(
            frame,
            "SPACE: new throw   R: reset   Q/ESC: quit",
            (WIDTH - 390, HEIGHT - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (50, 60, 65),
            1
        )

        return frame

    def run(self):
        while True:
            now = time.perf_counter()
            self.update(now)

            cv2.imshow(
                "CATCHBot Simulation",
                self.render()
            )

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