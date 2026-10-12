import tkinter as tk
from tkinter import ttk


class MyUserInterface:

    def __init__(
            self,
            max_value: int,
            process_name: str = "Progress Monitoring",
            title: str = "Progress Monitoring",
            geometry: str = "450x350",
    ) -> None:
        self.root = tk.Tk()

        self.root.title(process_name)
        self.root.geometry(geometry)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.value = 0
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

        self.info_label = ttk.Label(
            self.root,
            text="Ready"
        )
        self.info_label.pack(pady=10)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.root.destroy()

    def _update_progress(self, _value: int) -> None:
        self.value_label.config(
            text=f"Progress: {_value} from {self.max_value}"
        )
        self.progressbar["value"] = _value

    def update_ui(self, text: str, _value: int) -> None:
        self._update_progress(_value)
        self.info_label.config(text=text)

    def run_ui(self) -> None:
        self.root.mainloop()

    def on_close(self) -> None:
        self.root.destroy()
