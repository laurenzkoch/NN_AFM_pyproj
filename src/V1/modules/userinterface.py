import tkinter as tk
from tkinter import ttk
import queue
from typing import Callable

from V1.modules.eduvpn_processing import EDUVPNProcessor


class UserInterface:

    def __init__(
            self,
            root: tk.Tk | None = None,
            max_value: int = 100,
            process_name: str = "Progress Monitoring",
            title: str = "Progress Monitoring",
            geometry: str = "450x350",
            button_function: Callable[[], None] = lambda: None,
    ) -> None:

        if root is None:
            root = tk.Tk()

        self.root = root
        self.progress_queue: queue.Queue[tuple[str, bool | None]] = queue.Queue()
        self.button_function = button_function
        self.eduvpn_processor: EDUVPNProcessor | None = None
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
            root,
            text=title,
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=5)

        # Messwert
        self.value_label = ttk.Label(
            root,
            text=f"Progress: 0 from {max_value}",
            font=("Arial", 14)
        )
        self.value_label.pack(pady=10)

        # Fortschrittsanzeige
        self.progressbar = ttk.Progressbar(
            root,
            orient="horizontal",
            length=350,
            mode="determinate",
            maximum=self.max_value
        )
        self.progressbar.pack(pady=10)

        self.start_button_sim = ttk.Button(
            root,
            text="Start",
            command=lambda: self.button_function()
        )
        self.start_button_sim.pack(pady=15)

        self.points_label = ttk.Label(
            root,
            text="Ready"
        )
        self.points_label.pack(pady=10)


        status_frame = ttk.Frame(root)
        status_frame.pack(pady=20)

        self.start_button_edu = ttk.Button(
            status_frame,
            text="Start eduVPN",
            command=self.start_eduvpn
        )
        self.start_button_edu.pack(pady=15)

        ttk.Label(
            status_frame,
            text="eduVPN:"
        ).pack(side=tk.LEFT, padx=5)

        self.status_indicator = tk.Canvas(
            status_frame,
            width=30,
            height=30,
            highlightthickness=0
        )
        self.status_indicator.pack(side=tk.LEFT)

        self.status_led = self.status_indicator.create_oval(
            5,
            5,
            25,
            25,
            fill="red",
            outline="black"
        )


        self.poll_queue()

    def update_indicators(self, sim_points: int, dropped_points: int) -> None:

        value = sim_points + dropped_points

        self.value_label.config(
            text=f"Progress: {value} from {self.max_value}"
        )
        self.progressbar["value"] = value
        self.points_label.config(text=f"Dropped: {dropped_points}\nSimulated: {sim_points}")

    def poll_queue(self) -> None:
        try:
            while True:
                message_type, payload = self.progress_queue.get_nowait()

                match message_type:
                    case "progress":
                        if payload:
                            self.simulated_points += 1
                        else:
                            self.dropped_points += 1

                        self.update_indicators(
                            self.simulated_points,
                            self.dropped_points,
                        )
                    case "DONE":
                        self.points_label.config(
                            text=(
                                f"Dropped: {self.dropped_points}\n"
                                f"Simulated: {self.simulated_points}\n"
                                "Finished"
                            )
                        )
                        self.start_button_sim.config(state="normal", text="Start")
                        return
                    case "Starting MCAD":
                        self.points_label.config(text="Starting MCAD...")
                        self.start_button_sim.config(state="disabled", text="Running.")
                    case "Calculating":
                        self.points_label.config(text="Calculating...")

        except queue.Empty:
            pass

        self.root.after(100, self.poll_queue)

    def set_status(self, color: str) -> None:
        self.status_indicator.itemconfig(
            self.status_led,
            fill=color
        )

    def start_eduvpn(self) -> None:
        self.eduvpn_processor = EDUVPNProcessor()
        self.set_status("green")
        self.start_button_edu.config(state="disabled", text="Running.")

    def on_close(self):
        if self.eduvpn_processor is not None:
            try:
                self.eduvpn_processor.__exit__(None, None, None)
            except Exception:
                pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    UserInterface(root, max_value=1000)
    root.mainloop()


if __name__ == "__main__":
    main()