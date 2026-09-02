import tkinter as tk
from tkinter import ttk
from typing import Callable


class MyUserInterface:

    def __init__(
            self,
            max_value: int,
            process_name: str = "Progress Monitoring",
            title: str = "Progress Monitoring",
            geometry: str = "450x350",
            button_function: Callable[[], None] = lambda: None,
    ) -> None:

        self.root = tk.Tk()

        self.button_function = button_function
        self.root.title(process_name)
        self.root.geometry(geometry)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.running = False
        self.value = 0
        self.simulated_points = 0
        self.dropped_points = 0
        self.max_value = max_value


        # Überschrift
        title_label = ttk.Label(
            self.root,
            text=title,
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=5)

        # Messwert
        self.value_label = ttk.Label(
            self.root,
            text=f"Progress: 0 from {max_value}",
            font=("Arial", 14)
        )
        self.value_label.pack(pady=10)

        # Fortschrittsanzeige
        self.progressbar = ttk.Progressbar(
            self.root,
            orient="horizontal",
            length=350,
            mode="determinate",
            maximum=self.max_value
        )
        self.progressbar.pack(pady=10)

        self.start_button_sim = ttk.Button(
            self.root,
            text="Start",
            command=lambda: self.button_function()
        )
        self.start_button_sim.pack(pady=15)

        self.info_label = ttk.Label(
            self.root,
            text="Ready"
        )
        self.info_label.pack(pady=10)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.root.destroy()

    def update_progress(self, value: int) -> None:

        self.value_label.config(
            text=f"Progress: {value} from {self.max_value}"
        )
        self.progressbar["value"] = value


    def on_close(self) -> None:
        self.root.destroy()
