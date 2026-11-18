import math
import random
import time
from dataclasses import dataclass

import cv2
import numpy as np
from simulation_receiver import SimulationReceiver


# Window and layout settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 760
HEADER_BOTTOM = 190
FLOOR_Y = 620

# Trash-can settings
CAN_WIDTH = 112
CAN_HEIGHT = 105
CAN_SPEED = 430.0

# Simulated camera error
DETECTION_NOISE = 2.5


@dataclass
class Throw:
    """Stores the physics values for one thrown object."""

    x0: float
    y0: float
    vx: float
    vy: float
    gravity: float
    size: int = 17

    def position(self, elapsed):
        """Calculate the object's current x/y position."""

        x = self.x0 + self.vx * elapsed
        y = self.y0 + self.vy * elapsed
        y += 0.5 * self.gravity * elapsed**2

        return x, y

    def landing_time(self, target_y):
        """Calculate when the object reaches the catch line."""

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
    """Controls the throw, computer vision, prediction, and interface."""

    def __init__(self):
        self.random = random.Random()

        # Store live camera tracking history
        self.live_positions = []
        self.last_processed_track = 0.0

        # Receive external landing predictions
        self.receiver = SimulationReceiver()
        self.receiver.start()

        # Current trash-can position.
        self.can_x = 700.0

        # Current throw.
        self.throw = None

        # Time tracking for animation.
        self.throw_started = time.perf_counter()
        self.last_frame = time.perf_counter()

        # Stores:
        # time, center_x, center_y, box_width, box_height
        self.detections = []

        # Predicted and actual landing locations.
        self.predicted_x = None
        self.actual_x = None

        # Current simulation result.
        self.status = "READY"

        # Number of simulated camera detections.
        self.frames_seen = 0

        # Choose simulation mode
        self.mode = "LIVE"

        if self.mode == "SIMULATION":
            self.new_throw()

    def new_throw(self):
        """Automatically generate a throw from the left or right."""

        # Randomly choose a starting side.
        if self.random.choice(("left", "right")) == "left":
            start_x = self.random.uniform(220, 350)
            horizontal_velocity = self.random.uniform(230, 410)
        else:
            start_x = self.random.uniform(850, 980)
            horizontal_velocity = -self.random.uniform(230, 410)

        self.throw = Throw(
            x0=start_x,
            y0=self.random.uniform(330, 390),
            vx=horizontal_velocity,

            # These values keep the throw below the blue header.
            vy=self.random.uniform(-500, -390),

            gravity=self.random.uniform(980, 1160),
        )

        self.throw_started = time.perf_counter()
        self.detections.clear()
        self.predicted_x = None
        self.status = "THROWING"
        self.frames_seen = 0

        # Calculate the actual landing location for comparison.
        landing_time = self.throw.landing_time(FLOOR_Y - 14)

        if landing_time is not None:
            self.actual_x = (
                self.throw.x0
                + self.throw.vx * landing_time
            )
        else:
            self.actual_x = None

    def reset(self):
        """Reset the trash can and start another throw."""

        self.can_x = 700.0
        self.new_throw()

    def computer_vision_detection(self, true_x, true_y):
        """
        Simulate computer vision detecting the trash.

        In the real project, this function could be replaced by:
        - HSV color detection
        - YOLO object detection
        - another OpenCV detector
        """

        # Add a small amount of noise to imitate a real camera.
        center_x = true_x + self.random.gauss(0, DETECTION_NOISE)
        center_y = true_y + self.random.gauss(0, DETECTION_NOISE)

        # Simulated bounding-box size.
        box_width = self.throw.size * 2 + self.random.randint(-2, 3)
        box_height = self.throw.size * 2 + self.random.randint(-2, 3)

        return center_x, center_y, box_width, box_height

    def estimate_landing(self):
        """
        Estimate the landing point from detected center coordinates.

        Horizontal movement uses a line.
        Vertical movement uses a parabola.
        """

        # Do not estimate until enough frames are available.
        if len(self.detections) < 7:
            return

        recent = self.detections[-28:]

        times = np.array(
            [item[0] for item in recent],
            dtype=float
        )

        times -= times[0]

        center_x_values = np.array(
            [item[1] for item in recent],
            dtype=float
        )

        center_y_values = np.array(
            [item[2] for item in recent],
            dtype=float
        )

        # Estimate horizontal movement.
        x_fit = np.polyfit(
            times,
            center_x_values,
            1
        )

        x0 = x_fit[1]
        velocity_x = x_fit[0]

        # Estimate vertical movement.
        y_fit = np.polyfit(
            times,
            center_y_values,
            2
        )

        y0 = y_fit[2]
        velocity_y = y_fit[1]
        gravity = 2 * y_fit[0]

        if gravity <= 0:
            return

        # Calculate when the object reaches the floor.
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
        """Move the virtual trash can toward the prediction."""

        if target_x is None:
            return

        # Keep the trash can inside the window.
        minimum_x = CAN_WIDTH / 2
        maximum_x = WINDOW_WIDTH - CAN_WIDTH / 2

        target_x = max(
            minimum_x,
            min(maximum_x, target_x)
        )

        # Move gradually instead of teleporting.
        movement = CAN_SPEED * dt
        difference = target_x - self.can_x

        self.can_x += max(
            -movement,
            min(movement, difference)
        )

    
    def update_live_tracking(self):
        position = self.receiver.latest_position
        received_at = self.receiver.last_track_received
        now = time.monotonic()

        # Clear tracking after a longer detection gap
        if position is None or now - received_at > 1.5:
            self.live_positions.clear()
            return

        # Only process newly received coordinates
        if received_at <= self.last_processed_track:
            return

        # Start a new trail after a detection gap
        if (
            self.last_processed_track > 0
            and received_at - self.last_processed_track > 0.6
        ):
            self.live_positions.clear()

        self.last_processed_track = received_at

        # Convert camera coordinates to simulation coordinates
        x = position[0] * WINDOW_WIDTH / 700
        y = HEADER_BOTTOM + position[1] * (
            FLOOR_Y - HEADER_BOTTOM
        ) / 350

        point = (int(x), int(y))

        # Record actual received positions
        if not self.live_positions or self.live_positions[-1] != point:
            self.live_positions.append(point)

        # Keep more trajectory history
        self.live_positions = self.live_positions[-150:]


    def update(self, now):
        """Update physics, detection, prediction, and can movement."""

        dt = min(
            now - self.last_frame,
            0.05
        )

        self.last_frame = now

        #update live camera tracking
        self.update_live_tracking()

        # Live Camera Mode
        if self.mode == "LIVE":
            external_x = self.receiver.latest_target

            if (
                external_x is not None
                and time.monotonic() - self.receiver.last_received <= 0.5
            ):
                target_x = float(external_x) * WINDOW_WIDTH / 700
                self.move_trash_can(target_x, dt)

            return

        # Simulation Mode
        if self.status in ("CAUGHT", "MISSED"):
            return

        elapsed = now - self.throw_started
        true_x, true_y = self.throw.position(elapsed)

        # Continue tracking until the trash reaches the catch line.
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

            # Use frontend prediction when available
            external_x = self.receiver.latest_target

            if external_x is not None:
                print(f"Simulation received X: {external_x}")

            # Ignore predictions older than 0.5 seconds
            if time.monotonic() - self.receiver.last_received > 0.5:
                external_x = None

            if external_x is not None:
                # Convert dashboard coordinates (700px)
                # to simulation coordinates (1200px)
                target_x = float(external_x) * WINDOW_WIDTH/ 700
            else:
                # Keep Maria's original prediction
                target_x = self.predicted_x

            self.move_trash_can(target_x, dt)

        # Evaluate whether the trash can caught the object.
        else:
            catch_distance = CAN_WIDTH * 0.58

            if abs(self.can_x - true_x) <= catch_distance:
                self.status = "CAUGHT"
            else:
                self.status = "MISSED"

    def draw_room(self, frame):
        """Draw a simple uncluttered room."""

        # Plain wall below the header.
        cv2.rectangle(
            frame,
            (0, HEADER_BOTTOM),
            (WINDOW_WIDTH, 475),
            (211, 222, 226),
            -1
        )

        # Baseboard.
        cv2.line(
            frame,
            (0, 475),
            (WINDOW_WIDTH, 475),
            (125, 137, 140),
            5
        )

        cv2.line(
            frame,
            (0, 505),
            (WINDOW_WIDTH, 505),
            (185, 164, 143),
            12
        )

        # Floor.
        cv2.rectangle(
            frame,
            (0, 510),
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            (163, 128, 96),
            -1
        )

        # Floor perspective lines.
        for x in range(-200, WINDOW_WIDTH + 300, 75):
            cv2.line(
                frame,
                (x, 510),
                (x + 300, WINDOW_HEIGHT),
                (184, 148, 112),
                2
            )

        # Window.
        cv2.rectangle(
            frame,
            (70, 235),
            (300, 405),
            (82, 105, 120),
            -1
        )

        cv2.rectangle(
            frame,
            (82, 247),
            (288, 393),
            (150, 206, 225),
            -1
        )

        cv2.line(
            frame,
            (185, 247),
            (185, 393),
            (235, 240, 240),
            5
        )

        cv2.line(
            frame,
            (82, 320),
            (288, 320),
            (235, 240, 240),
            5
        )

        cv2.putText(
            frame,
            "ROOM CAMERA VIEW",
            (78, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (70, 80, 84),
            1
        )

        # Open rug/catch zone.
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
        """Draw the fixed camera on a tripod."""

        camera_x = 825
        camera_y = 285

        # Tripod legs.
        cv2.line(
            frame,
            (camera_x, camera_y + 40),
            (camera_x - 120, 500),
            (48, 48, 48),
            5
        )

        cv2.line(
            frame,
            (camera_x, camera_y + 40),
            (camera_x + 100, 500),
            (48, 48, 48),
            5
        )

        cv2.line(
            frame,
            (camera_x, camera_y + 40),
            (camera_x, 500),
            (48, 48, 48),
            5
        )

        # Camera body.
        cv2.rectangle(
            frame,
            (camera_x - 52, camera_y - 28),
            (camera_x + 52, camera_y + 35),
            (32, 37, 40),
            -1
        )

        # Camera lens.
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
            "FIXED TRIPOD CAMERA",
            (705, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (45, 55, 60),
            2
        )

    def draw_trash_can(self, frame):
        """Draw the mobile trash can."""

        left = int(
            self.can_x - CAN_WIDTH / 2
        )

        right = int(
            self.can_x + CAN_WIDTH / 2
        )

        top = FLOOR_Y - CAN_HEIGHT

        # Lid.
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

        # Body.
        cv2.rectangle(
            frame,
            (left, top),
            (right, FLOOR_Y),
            (63, 84, 94),
            -1
        )

        # Bottom rim.
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

        # Lid handle.
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

        # Green light means the object was caught.
        if self.status == "CAUGHT":
            cv2.circle(
                frame,
                (int(self.can_x), top + 25),
                12,
                (30, 220, 80),
                -1
            )

    def draw_detection_overlay(self, frame):
        """Draw the bounding box, center point, and trajectory trace."""

        if not self.detections:
            return

        # Prevent the trajectory from appearing over the blue header.
        room_detections = [
            row
            for row in self.detections
            if HEADER_BOTTOM <= row[2] <= FLOOR_Y
        ]

        if not room_detections:
            return

        # Purple line represents the detected path.
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

        # Get the newest bounding box.
        _, center_x, center_y, box_width, box_height = (
            room_detections[-1]
        )

        x1 = int(center_x - box_width / 2)
        y1 = int(center_y - box_height / 2)
        x2 = int(center_x + box_width / 2)
        y2 = int(center_y + box_height / 2)

        # Green box represents the computer-vision detection.
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (40, 230, 70),
            3
        )

        # Yellow dot represents the detected center.
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

    
    def draw_live_tracking(self, frame):
        # Only display recent live camera tracking
        if (
            not self.live_positions
            or time.monotonic() - self.receiver.last_track_received > 1.5
        ):
            return

        # Draw purple tracking trajectory
        points = np.array(self.live_positions, dtype=np.int32)

        if len(points) > 1:
            cv2.polylines(
                frame, [points], False,
                (255, 80, 210), 3
            )

        # Draw green square around current object position
        x, y = self.live_positions[-1]

        cv2.rectangle(
            frame,
            (x - 20, y - 20),
            (x + 20, y + 20),
            (40, 230, 70), 3
        )

        cv2.putText(
            frame,
            "LIVE CAMERA OBJECT",
            (max(5, x - 80), max(HEADER_BOTTOM + 20, y - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (40, 230, 70), 2
        )


    def render(self):
        """Create the complete graphical interface."""

        # Blue gradient workspace.
        frame = np.zeros(
            (WINDOW_HEIGHT, WINDOW_WIDTH, 3),
            dtype=np.uint8
        )

        for y in range(WINDOW_HEIGHT):
            color = int(235 - 35 * y / WINDOW_HEIGHT)
            frame[y, :] = (color, 244, 250)

        # Blue CATCHBot header.
        cv2.rectangle(
            frame,
            (0, 0),
            (WINDOW_WIDTH, HEADER_BOTTOM),
            (180, 90, 30),
            -1
        )

        cv2.putText(
            frame,
            "CATCHBot",
            (25, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.92,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "COMPUTER-VISION THROW SIMULATION",
            (190, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (225, 248, 255),
            1
        )

        cv2.putText(
            frame,
            "Fixed camera  |  Detect  |  Track  |  Predict  |  Catch",
            (27, 69),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 246, 255),
            1
        )

        # Current values used by the top cards.
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
            (70, 220, 105)
            if self.status == "CAUGHT"
            else (255, 255, 255)
        )

        # Live Tracking blue card.
        cv2.rectangle(
            frame,
            (20, 83),
            (510, 174),
            (42, 93, 125),
            -1
        )

        cv2.rectangle(
            frame,
            (20, 83),
            (510, 174),
            (190, 235, 250),
            2
        )

        cv2.putText(
            frame,
            "LIVE TRACKING",
            (38, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"STATUS: {self.status}",
            (38, 134),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            status_color,
            2
        )

        cv2.putText(
            frame,
            f"Frames: {self.frames_seen}",
            (190, 134),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 250, 255),
            1
        )

        cv2.putText(
            frame,
            f"Predicted X: {predicted}",
            (300, 134),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 250, 255),
            1
        )

        cv2.putText(
            frame,
            f"Actual X: {actual}   Can X: {self.can_x:.0f}px",
            (38, 158),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 242, 250),
            1
        )

        # Simulation Controls blue card.
        cv2.rectangle(
            frame,
            (530, 83),
            (1180, 174),
            (42, 93, 125),
            -1
        )

        cv2.rectangle(
            frame,
            (530, 83),
            (1180, 174),
            (190, 235, 250),
            2
        )

        cv2.putText(
            frame,
            "SIMULATION CONTROLS",
            (550, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Random throw source:",
            (550, 136),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (225, 245, 252),
            1
        )

        cv2.putText(
            frame,
            "LEFT / RIGHT",
            (705, 136),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "SPACE = new throw    R = reset    Q/ESC = quit",
            (550, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (225, 245, 252),
            1
        )

        # Draw the room below the header.
        self.draw_room(frame)

        true_x, true_y = 0, 0

        if self.mode == "SIMULATION":
            elapsed = time.perf_counter() - self.throw_started
            true_x, true_y = self.throw.position(elapsed)

        # Red thrown object
        if (
            self.mode == "SIMULATION"
            and self.status == "THROWING"
            and HEADER_BOTTOM + self.throw.size <= true_y <= FLOOR_Y
        ):
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

        if self.mode == "SIMULATION":
            self.draw_detection_overlay(frame)
        else:
            self.draw_live_tracking(frame)

        # Orange predicted landing line.
        if (
            self.mode == "SIMULATION"
            and self.predicted_x is not None
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

        # Draw the trash can and fixed tripod camera.
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
            if self.mode == "SIMULATION" and self.status == "CAUGHT"
            else (45, 55, 60)
        )


        if self.mode == "SIMULATION" and self.status == "CAUGHT":
            cv2.putText(
                frame,
                "TRASH CAUGHT!",
                (475, 390),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.15,
                (30, 190, 70),
                3
            )

        elif self.mode == "SIMULATION" and self.status == "MISSED":
            cv2.putText(
                frame,
                "THROW MISSED",
                (490, 390),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (30, 50, 210),
                3
            )

        # Bottom legend and keyboard controls.
        cv2.rectangle(
            frame,
            (0, WINDOW_HEIGHT - 47),
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            (23, 47, 70),
            -1
        )

        cv2.putText(
            frame,
            "GREEN BOX = DETECTION",
            (25, WINDOW_HEIGHT - 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (115, 255, 125),
            1
        )

        cv2.putText(
            frame,
            "PURPLE = TRACK",
            (260, WINDOW_HEIGHT - 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 130, 225),
            1
        )

        cv2.putText(
            frame,
            "ORANGE = PREDICTION",
            (430, WINDOW_HEIGHT - 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (80, 180, 255),
            1
        )

        cv2.putText(
            frame,
            "SPACE: New throw   R: Reset   Q/ESC: Quit",
            (780, WINDOW_HEIGHT - 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (220, 240, 248),
            1
        )

        return frame

    def run(self):
        """Start the OpenCV window and keyboard controls."""

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

        self.receiver.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    CatchBotSimulation().run()