"""Interactive trash-can movement simulation.

Run with:
    python trash_can_simulation.py

Controls:
    R / Right Arrow  - turn right
    L / Left Arrow   - turn left
    S / Space        - stop or resume
    T                - return to automatic target tracking
    Mouse click      - choose a new target position
"""

import math
import tkinter as tk


WIDTH, HEIGHT = 900, 560
CANVAS_BG = "#eef4f7"


class TrashCanSimulation:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Autonomous Trash Can Simulation")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT,
                                bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack()

        self.x, self.y = 150.0, 390.0
        self.angle = -35.0       # Degrees; 0 points right.
        self.target_x, self.target_y = 720.0, 190.0
        self.speed = 2.8
        self.turn_speed = 4.5
        self.moving = True
        self.manual_turn = 0     # -1 = left, 1 = right, 0 = auto steering
        self.frame = 0

        self.canvas.bind("<Button-1>", self.choose_target)
        root.bind("<KeyPress>", self.handle_key)
        root.focus_force()

        self.draw_scene()
        self.update()

    def choose_target(self, event):
        """Set a new target wherever the user clicks."""
        self.target_x, self.target_y = event.x, event.y
        self.moving = True
        self.manual_turn = 0

    def handle_key(self, event):
        key = event.keysym.lower()
        if key in ("r", "right"):
            self.manual_turn = 1
            self.moving = True
        elif key in ("l", "left"):
            self.manual_turn = -1
            self.moving = True
        elif key in ("s", "space"):
            self.moving = not self.moving
            self.manual_turn = 0
        elif key == "t":
            self.manual_turn = 0
            self.moving = True

    def normalize_angle(self, degrees):
        return (degrees + 180) % 360 - 180

    def steer_toward_target(self):
        """Rotate toward the target using the shortest angular direction."""
        desired = math.degrees(math.atan2(self.target_y - self.y,
                                          self.target_x - self.x))
        difference = self.normalize_angle(desired - self.angle)
        self.angle += max(-self.turn_speed, min(self.turn_speed, difference))

    def update(self):
        if self.moving:
            if self.manual_turn:
                self.angle += self.manual_turn * self.turn_speed
            else:
                self.steer_toward_target()

            radians = math.radians(self.angle)
            self.x += math.cos(radians) * self.speed
            self.y += math.sin(radians) * self.speed

            # Keep the trash can inside the simulation area.
            self.x = max(45, min(WIDTH - 45, self.x))
            self.y = max(120, min(HEIGHT - 55, self.y))

        self.frame += 1
        self.draw_scene()
        self.root.after(25, self.update)
    def draw_scene(self):
        self.canvas.delete("all")

        # Header and instructions.
        self.canvas.create_rectangle(0, 0, WIDTH, 92, fill="#17324d", outline="")
        self.canvas.create_text(25, 25, anchor="w", text="AUTONOMOUS TRASH CAN",
                               fill="white", font=("Arial", 20, "bold"))
        self.canvas.create_text(
            25, 59, anchor="w",
            text="Click to set a target  •  R/right = turn right  •  L/left = turn left  •  S/space = stop/resume",
            fill="#cfe5f2", font=("Arial", 11))

        # Target marker and a dashed line to it.
        self.canvas.create_oval(self.target_x - 16, self.target_y - 16,
                                self.target_x + 16, self.target_y + 16,
                                outline="#e56b45", width=3)
        self.canvas.create_line(self.target_x - 9, self.target_y,
                                self.target_x + 9, self.target_y,
                                fill="#e56b45", width=2)
        self.canvas.create_line(self.target_x, self.target_y - 9,
                                self.target_x, self.target_y + 9,
                                fill="#e56b45", width=2)
        self.canvas.create_text(self.target_x + 22, self.target_y - 19,
                               text="TARGET", fill="#b94c2c",
                               font=("Arial", 10, "bold"))
        self.canvas.create_line(self.x, self.y, self.target_x, self.target_y,
                               fill="#b8cbd4", dash=(5, 5))

        self.draw_trash_can()

        status = "MOVING" if self.moving else "STOPPED"
        mode = "MANUAL TURN" if self.manual_turn else "AUTO TARGET"
        distance = math.hypot(self.target_x - self.x, self.target_y - self.y)
        info = (f"Status: {status}    Mode: {mode}    "
                f"Position: ({self.x:.0f}, {self.y:.0f})    "
                f"Angle: {self.angle % 360:.0f}°    "
                f"Target distance: {distance:.0f}px")
        self.canvas.create_rectangle(0, HEIGHT - 42, WIDTH, HEIGHT,
                                     fill="#17324d", outline="")
        self.canvas.create_text(20, HEIGHT - 21, anchor="w", text=info,
                               fill="white", font=("Consolas", 11))

    def draw_trash_can(self):
        """Draw the can around its center and rotate its directional arrow."""
        # The body is intentionally upright while the arrow shows heading.
        left, top, right, bottom = self.x - 31, self.y - 34, self.x + 31, self.y + 38
        self.canvas.create_oval(left - 3, bottom - 8, right + 3, bottom + 7,
                                fill="#132333", outline="")
        self.canvas.create_polygon(left, top + 8, right, top + 8,
                                   right - 7, bottom, left + 7, bottom,
                                   fill="#4d6876", outline="#233947", width=2)
        self.canvas.create_oval(left - 5, top - 7, right + 5, top + 11,
                                fill="#7f9baa", outline="#233947", width=2)
        self.canvas.create_oval(self.x - 6, self.y - 3, self.x + 6, self.y + 9,
                                fill="#263f4d", outline="")

        radians = math.radians(self.angle)
        tip_x = self.x + math.cos(radians) * 55
        tip_y = self.y + math.sin(radians) * 55
        self.canvas.create_line(self.x, self.y + 3, tip_x, tip_y,
                                fill="#e56b45", width=3, arrow=tk.LAST)


if __name__ == "__main__":
    app = tk.Tk()
    TrashCanSimulation(app)
    app.mainloop()

        
       