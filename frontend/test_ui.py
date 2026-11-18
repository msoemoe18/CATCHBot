
import tkinter as tk

root = tk.Tk()
root.title("CATCHBot Display Test")
root.geometry("600x400")
root.configure(bg="white")

title = tk.Label(
    root,
    text="CATCHBot Dashboard",
    font=("Arial", 24, "bold"),
    bg="white",
    fg="black"
)
title.pack(pady=30)

status = tk.Label(
    root,
    text="Camera Status: Waiting",
    font=("Arial", 16),
    bg="white",
    fg="blue"
)
status.pack(pady=20)

button = tk.Button(
    root,
    text="Test Button",
    command=lambda: status.config(text="Button Works!")
)
button.pack(pady=20)

root.mainloop()
