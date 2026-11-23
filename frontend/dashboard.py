
import tkinter as tk
import cv2
import sys
import time
import math
import numpy as np
import socket
import json

from pathlib import Path
from queue import Empty
from PIL import Image, ImageTk

from trajectory_view import draw_trajectory
from udp_listener import UDPListener

# Import Angie's vision modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "vision"))

from detector_yolo import YoloDetector
from tracker import Tracker
from kalman import BallKalman
from trajectory import landing_from_state

# -----------------------------------
# Dashboard Setup
# -----------------------------------

root = tk.Tk()
root.tk.call("tk", "scaling", 1.0)
root.title("CATCHBot - AI Trash Can")
# Dashboard window: left side
root.geometry("565x760+0+35")
root.minsize(500, 650)
root.configure(bg="#142235")

camera = None
camera_running = False
camera_job = None
udp_job = None

detector = None
tracker = Tracker(max_points=30, max_missing=5)
kalman_filter = None
last_frame_time = None
last_radius = 10

# Tracking visualization settings
MIN_MOVEMENT = 12
SMOOTHING_FACTOR = 0.35
MAX_TRAIL_POINTS = 20

smooth_position = None
last_tracked_position = None
smooth_trail = []

# Experimental prediction settings
MIN_PREDICTION_SPEED = 100  # pixels/second
MIN_TRACK_POINTS = 5
PREDICTION_SECONDS = 0.5

# Experimental landing prediction settings
CATCH_HEIGHT_Y = 330
TEST_GRAVITY_PX = 500.0

# UDP communication
udp_listener = UDPListener()
latest_landing = None
last_landing_time = 0.0
LANDING_DISPLAY_SECONDS = 0.5
last_coordinates_time = 0.0
COORDINATES_DISPLAY_SECONDS = 0.5

# Send landing predictions to Maria's simulation
simulation_socket = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)

SIMULATION_ADDRESS = ("127.0.0.1", 5006)

# -----------------------------------
# Dashboard Header
# -----------------------------------

title = tk.Label(
    root,
    text="CATCHBot Dashboard",
    font=("Arial", 26, "bold"),
    fg="white",
    bg="#142235"
)
title.pack(pady=(20, 5))

subtitle = tk.Label(
    root,
    text="AI-Powered Autonomous Trash Can",
    font=("Arial", 13),
    fg="#94A3B8",
    bg="#142235"
)
subtitle.pack(pady=(0, 15))

# -----------------------------------
# Camera Display
# -----------------------------------

camera_frame = tk.Frame(
    root,
    bg="#26384C",
    width=545,
    height=270
)
camera_frame.pack(pady=15)
camera_frame.pack_propagate(False)

camera_label = tk.Label(
    camera_frame,
    text="Camera Offline",
    font=("Arial", 16),
    fg="white",
    bg="#26384C"
)
camera_label.pack(expand=True, fill="both")

# -----------------------------------
# Status Labels
# -----------------------------------

status_label = tk.Label(
    root,
    text="Camera Status: Offline",
    font=("Arial", 14),
    fg="#94A3B8",
    bg="#142235"
)
status_label.pack(pady=10)

coordinates_label = tk.Label(
    root,
    text="Object Coordinates: N/A",
    font=("Arial", 14),
    fg="white",
    bg="#142235"
)
coordinates_label.pack(pady=5)

prediction_label = tk.Label(
    root,
    text="Predicted Landing Position: N/A",
    font=("Arial", 14),
    fg="white",
    bg="#142235"
)
prediction_label.pack(pady=5)

# -----------------------------------
# UDP Receiver
# -----------------------------------

def check_udp_messages():
    global latest_landing, udp_job

    try:
        while True:
            message = udp_listener.messages.get_nowait()

            if camera_running and message.get("type") == "landing":
                x_land = message.get("x_land")

                if x_land is not None:
                    latest_landing = float(x_land)

                    prediction_label.config(
                        text=(
                            f"UDP Landing X: "
                            f"{latest_landing:.1f} px"
                        )
                    )

    except Empty:
        pass

    udp_job = root.after(50, check_udp_messages)


# -----------------------------------
# Live Camera and Prediction
# -----------------------------------

def update_camera():
    global camera_job
    global kalman_filter, last_frame_time, last_radius
    global smooth_position, last_tracked_position
    global last_landing_time
    global last_coordinates_time

    if not camera_running or camera is None:
        return

    success, frame = camera.read()

    if not success:
        stop_camera()
        status_label.config(
            text="Camera Status: Error",
            fg="#F87171"
        )
        return

    # Resize camera frame
    frame = cv2.resize(frame, (700, 350))

    # Flip camera horizontally (mirror view)
    frame = cv2.flip(frame, 1)

    current_time = time.perf_counter()

    if last_frame_time is None:
        dt = 1 / 30
    else:
        dt = max(
            0.001,
            min(current_time - last_frame_time, 0.2)
        )

    last_frame_time = current_time

    # Real YOLO detection
    detection = detector.detect(frame)

    # Angie's object tracker
    tracker.update(detection, current_time)

    predicted_path = None
    predicted_landing = None
    bbox = None

    #prediction_label.config(
    #    text="Predicted Landing Position: N/A"
    #)

    # Use the experimental gravity setting defined above
    calibrated_radius = TEST_GRAVITY_PX * 0.02 / 9.81

    if kalman_filter is not None:
        kalman_filter.predict(dt, calibrated_radius)

    if detection is not None:
        x, y, radius = detection

        # Send live object coordinates to Maria's simulation
        track_message = {
            "type": "track",
            "x": float(x),
            "y": float(y)
        }

        simulation_socket.sendto(
            json.dumps(track_message).encode("utf-8"),
            SIMULATION_ADDRESS
        )

        last_radius = radius

        if kalman_filter is None:
            kalman_filter = BallKalman(x, y)
        else:
            kalman_filter.update(x, y)

        # Smooth tracking position
        if smooth_position is None:
            smooth_position = (float(x), float(y))
        else:
            old_x, old_y = smooth_position

            smooth_position = (
                SMOOTHING_FACTOR * x
                + (1 - SMOOTHING_FACTOR) * old_x,
                SMOOTHING_FACTOR * y
                + (1 - SMOOTHING_FACTOR) * old_y
            )

        current_x = int(smooth_position[0])
        current_y = int(smooth_position[1])

        # Add a point only after meaningful movement
        if last_tracked_position is None:
            smooth_trail.append((current_x, current_y))
            last_tracked_position = (current_x, current_y)
        else:
            distance = math.hypot(
                current_x - last_tracked_position[0],
                current_y - last_tracked_position[1]
            )

            if distance >= MIN_MOVEMENT:
                smooth_trail.append((current_x, current_y))
                last_tracked_position = (current_x, current_y)

        if len(smooth_trail) > MAX_TRAIL_POINTS:
            smooth_trail.pop(0)

        bbox = (
            x - radius,
            y - radius,
            radius * 2,
            radius * 2
        )

        last_coordinates_time = time.monotonic()

        coordinates_label.config(
            text=f"Object Coordinates: ({x}, {y})"
        )

        status_label.config(
            text=(
                f"Detected: {detector.last_label} "
                f"({detector.last_confidence:.0%})"
            ),
            fg="#4ADE80"
        )

        # Experimental Kalman future prediction
        if len(tracker.points) >= MIN_TRACK_POINTS:
            kx, ky, vx, vy = kalman_filter.kf.x
            speed = math.hypot(vx, vy)

            print(
                f"Velocity X: {vx:.1f} px/s | "
                f"Velocity Y: {vy:.1f} px/s | "
                f"Speed: {speed:.1f} px/s"
            )
           
            if speed >= MIN_PREDICTION_SPEED:
                times = np.linspace(
                    0, PREDICTION_SECONDS, 30
                )

                future = np.column_stack([
                    kx + vx * times,
                    ky + vy * times
                    + 0.5 * TEST_GRAVITY_PX * times**2
                ])

                
                if np.isfinite(future).all():
                    predicted_path = future.tolist()

                    result = landing_from_state(
                        kx, ky, vx, vy,
                        TEST_GRAVITY_PX, CATCH_HEIGHT_Y
                    )

                    if result is not None:
                        x_land, t_land = result

                        if (np.isfinite(x_land)
                                and 0 <= x_land < 700
                                and 0 < t_land <= 2):
                            predicted_landing = (
                                int(x_land), CATCH_HEIGHT_Y
                            )

                            times = np.linspace(0, t_land, 40)
                            predicted_path = np.column_stack([
                                kx + vx * times,
                                ky + vy * times
                                + 0.5 * TEST_GRAVITY_PX * times**2
                            ]).tolist()

                            last_landing_time = time.monotonic()

                            prediction_label.config(
                                text=f"Experimental Landing: ({int(x_land)}, {CATCH_HEIGHT_Y})"
                            )

                            # Send predicted landing X to Maria's simulation
                            message = {
                                "type": "landing",
                                "x_land": float(x_land)
                            }

                            simulation_socket.sendto(
                                json.dumps(message).encode("utf-8"),
                                SIMULATION_ADDRESS
                            )
                            print(f"Sent landing X to simulation: {x_land:.1f}")

    else:
        status_label.config(
            text="Camera Active - No Object Detected",
            fg="#FACC15"
        )

        #coordinates_label.config(
        #    text="Object Coordinates: N/A"
        #)

        if (
            time.monotonic() - last_coordinates_time
            > COORDINATES_DISPLAY_SECONDS
        ):
            coordinates_label.config(
                text="Object Coordinates: N/A"
            )

        if tracker.missing > tracker.max_missing:
            smooth_trail.clear()
            smooth_position = None
            last_tracked_position = None
            kalman_filter = None

    # Clear the displayed prediction after 2 seconds without an update
        if (
            last_landing_time > 0
            and time.monotonic() - last_landing_time > LANDING_DISPLAY_SECONDS
        ):
            prediction_label.config(
                text="Predicted Landing Position: N/A"
            )
            last_landing_time = 0.0

    # Draw green trail and experimental red path
    frame = draw_trajectory(
        frame,
        positions=smooth_trail,
        bbox=bbox,
        predicted_path=predicted_path,
        predicted_landing=predicted_landing
    )

    # Convert frame for Tkinter
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(frame)

    # Resize camera preview without changing detection coordinates
    image.thumbnail((535, 300), Image.Resampling.LANCZOS)

    photo = ImageTk.PhotoImage(image=image)

    camera_label.config(image=photo, text="")
    camera_label.image = photo

    camera_job = root.after(30, update_camera)


# -----------------------------------
# Camera Controls
# -----------------------------------

def start_camera():
    global camera, camera_running, detector
    global kalman_filter, last_frame_time
    global smooth_position, last_tracked_position

    if camera_running:
        return

    camera = cv2.VideoCapture(
        0, cv2.CAP_AVFOUNDATION
    )

    if not camera.isOpened():
        camera.release()
        camera = None

        status_label.config(
            text="Camera Status: Unable to Open",
            fg="#F87171"
        )
        return

    if detector is None:
        status_label.config(
            text="Loading YOLO model...",
            fg="#FACC15"
        )
        root.update_idletasks()

        try:
            detector = YoloDetector()
        except Exception as error:
            camera.release()
            camera = None
            status_label.config(
                text=f"YOLO Error: {error}",
                fg="#F87171"
            )
            return

    tracker.points.clear()
    tracker.missing = 0

    smooth_trail.clear()
    smooth_position = None
    last_tracked_position = None
    kalman_filter = None
    last_frame_time = None

    camera_running = True
    update_camera()


def stop_camera():
    global camera, camera_running, camera_job
    global kalman_filter, last_frame_time
    global smooth_position, last_tracked_position
    global latest_landing
    global last_landing_time
    global last_coordinates_time

    latest_landing = None
    last_landing_time = 0.0
    last_coordinates_time = 0.0

    camera_running = False

    if camera_job is not None:
        root.after_cancel(camera_job)
        camera_job = None

    if camera is not None:
        camera.release()
        camera = None

    tracker.points.clear()
    tracker.missing = 0

    smooth_trail.clear()
    smooth_position = None
    last_tracked_position = None
    kalman_filter = None
    last_frame_time = None

    camera_label.config(
        image="",
        text="Camera Offline"
    )
    camera_label.image = None

    status_label.config(
        text="Camera Status: Offline",
        fg="#94A3B8"
    )

    coordinates_label.config(
        text="Object Coordinates: N/A"
    )

    prediction_label.config(
    text="Predicted Landing Position: N/A",
    fg="white"
)


def on_close():
    global udp_job

    stop_camera()

    if udp_job is not None:
        root.after_cancel(udp_job)
        udp_job = None

    udp_listener.stop()
    simulation_socket.close()
    root.destroy()


# -----------------------------------
# Start and Stop Buttons
# -----------------------------------

button_frame = tk.Frame(root, bg="#142235")
button_frame.pack(pady=20)

start_button = tk.Button(
    button_frame,
    text="Start Camera",
    command=start_camera,
    font=("Arial", 12, "bold"),
    width=15,
    relief="raised"
)
start_button.pack(side="left", padx=10)

stop_button = tk.Button(
    button_frame,
    text="Stop Camera",
    command=stop_camera,
    font=("Arial", 12, "bold"),
    width=15,
    relief="raised"
)
stop_button.pack(side="left", padx=10)

# -----------------------------------
# Start UDP Communication
# -----------------------------------

try:
    udp_listener.start()
    check_udp_messages()
except OSError as error:
    prediction_label.config(
        text=f"UDP Connection Error: {error}",
        fg="#F87171"
    )

root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()
