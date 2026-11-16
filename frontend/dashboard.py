import tkinter as tk
import cv2
from PIL import Image, ImageTk
from trajectory_view import draw_trajectory

# Create main window
root = tk.Tk()
root.tk.call("tk", "scaling", 1.0)
root.title("CATCHBot - AI Trash Can")
root.geometry("1000x750")
root.configure(bg="#142235")

# Dashboard title
title = tk.Label(
    root,
    text="CATCHBot Dashboard",
    font=("Arial", 26, "bold"),
    fg="white",
    bg="#142235"
)
title.pack(pady=(20, 5))

# Subtitle
subtitle = tk.Label(
    root,
    text="AI-Powered Autonomous Trash Can",
    font=("Arial", 13),
    fg="#94A3B8",
    bg="#142235"
)
subtitle.pack(pady=(0,15))

# Camera placeholder
camera_frame = tk.Frame(
    root,
    bg="#26384C",
    width=700,
    height=350
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
camera_label.pack(expand=True)

camera = None
camera_running = False

# Detection status
status_label = tk.Label(
    root,
    text="Detection Status: Waiting",
    font=("Arial", 14),
    fg="#4ADE80",
    bg="#142235"
)
status_label.pack(pady=10)

# Object coordinates
coordinates_label = tk.Label(
    root,
    text="Object Coordinates: N/A",
    font=("Arial", 14),
    fg="white",
    bg="#142235"
)
coordinates_label.pack(pady=5)

# Predicted landing position
prediction_label = tk.Label(
    root,
    text="Predicted Landing Position: N/A",
    font=("Arial", 14),
    fg="white",
    bg="#142235"
)
prediction_label.pack(pady=5)


def update_camera():
    
    if not camera_running or camera is None:
        return

    success, frame = camera.read()

    if success:

        # Temporary sample coordinates
        # Replace with Angie's backend data later
        sample_positions = [
            (100, 100),
            (150, 120),
            (200, 150),
            (250, 190),
            (300, 240)
        ]

        sample_bbox = (280, 220, 40, 40)
        sample_landing = (450, 320)

        # Draw tracking points, trajectory,
        # bounding box and landing marker
        frame = draw_trajectory(
            frame,
            positions=sample_positions,
            bbox=sample_bbox,
            predicted_landing=sample_landing
        )

        # Convert BGR to RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Resize for dashboard
        frame = cv2.resize(
            frame,
            (700, 350)
        )

        # Convert to Tkinter image
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)

        # Display camera feed
        camera_label.config(image=photo, text="")
        camera_label.image = photo

        # Update dashboard information
        status_label.config(
            text="Camera Active - Demo Trajectory",
            fg="#4ADE80"
        )

        coordinates_label.config(
            text="Object Coordinates: (300, 240) [Demo]"
        )

        prediction_label.config(
            text="Predicted Landing Position: (450, 320) [Demo]"
        )

        # Refresh camera
        root.after(30, update_camera)
    else:
        stop_camera()
        status_label.config(
            text="Detection Status: Camera Error",
            fg="#F87171"
        )


def start_camera():
    global camera, camera_running

    if camera_running:
        return

    camera = cv2.VideoCapture(
        0, 
        cv2.CAP_AVFOUNDATION
    )

    if not camera.isOpened():
        camera.release()
        camera = None

        status_label.config(
            text="Camera Status: Unable to Open",
            fg="#F87171"
        )
        return

    camera_running = True

    status_label.config(
        text="Camera Active - Detection Not Connected",
        fg="#4ADE80"
    )

    start_button.config(state="disabled")
    stop_button.config(state="normal")
    #start_button.pack_forget()
    #stop_button.pack(side="left", padx=10)
    update_camera()


def stop_camera():
    global camera, camera_running

    camera_running = False

    if camera is not None:
        camera.release()
        camera = None

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
        text="Predicted Landing Position: N/A"
    )

    start_button.config(state="normal")
    stop_button.config(state="disabled")
    #stop_button.pack_forget()
    #start_button.pack(side="left", padx=10)

def on_close():
    stop_camera()
    root.destroy()


button_frame = tk.Frame(
    root,
    bg="#142235"
)
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
    relief="raised",
    #state="disabled"
)
stop_button.pack(side="left", padx=10)

# Release camera when window closes
root.protocol("WM_DELETE_WINDOW", on_close)

# Run the application
root.mainloop()


