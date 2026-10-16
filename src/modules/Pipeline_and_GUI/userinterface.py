import tkinter as tk
from tkinter import ttk
from types import TracebackType


class MyUserInterface:
    def __init__(
        self,
        max_value: int,
        process_name: str = "Progress Monitoring",
        title: str = "Progress Monitoring",
        geometry: str = "450x350",
        overall_max_value: int | None = None,
    ) -> None:
        self.root = tk.Tk()

        self.root.title(process_name)
        self.root.geometry(geometry)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.value = 0
        self.close_requested = False
        self.service_finished = False

        self.max_value = max_value

        # Überschrift
        title_label = ttk.Label(self.root, text=title, font=("Arial", 18, "bold"))
        title_label.pack(pady=5)

        self.overall_max_value: int | None = overall_max_value
        self.overall_value_label: ttk.Label | None = None
        self.overall_progressbar: ttk.Progressbar | None = None

        if overall_max_value is not None:
            if overall_max_value < 1:
                raise ValueError("overall_max_value must be at least 1.")

            self.overall_value_label = ttk.Label(
                self.root,
                text=f"Files: 0 from {overall_max_value}",
                font=("Arial", 14),
            )
            self.overall_value_label.pack(pady=5)

            self.overall_progressbar = ttk.Progressbar(
                self.root,
                orient="horizontal",
                length=350,
                mode="determinate",
                maximum=overall_max_value,
            )
            self.overall_progressbar.pack(pady=5)

        # Messwert
        self.value_label = ttk.Label(
            self.root, text=f"Progress: 0 from {max_value}", font=("Arial", 14)
        )
        self.value_label.pack(pady=10)

        # Fortschrittsanzeige
        self.progressbar = ttk.Progressbar(
            self.root,
            orient="horizontal",
            length=350,
            mode="determinate",
            maximum=self.max_value,
        )
        self.progressbar.pack(pady=10)

        self.info_label = ttk.Label(self.root, text="Ready")
        self.info_label.pack(pady=10)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.root.destroy()

    def update_ui(
        self,
        text: str,
        _value: int,
        *,
        max_steps: int | None = None,
        overall_value: int | None = None,
    ) -> None:
        if max_steps is not None:
            if max_steps < 1:
                raise ValueError("max_steps must be at least 1.")

            self.max_value = max_steps
            self.progressbar.configure(maximum=max_steps)

        self._update_progress(_value)
        self.info_label.config(text=text)

        if (
            overall_value is not None
            and self.overall_value_label is not None
            and self.overall_progressbar is not None
        ):
            self.overall_value_label.config(
                text=(f"Files: {overall_value} from {self.overall_max_value}")
            )
            self.overall_progressbar["value"] = overall_value

    def run_ui(self) -> None:
        self.root.mainloop()

    def on_close(self) -> None:
        self.close_requested = True
        if self.service_finished:
            self.root.quit()
        else:
            # Keep Tk's event loop alive while background work finishes.
            self.root.withdraw()

    def mark_finished(self) -> None:
        self.service_finished = True
        if self.close_requested:
            self.root.quit()

    def destroy(self) -> None:
        if self.root.winfo_exists():
            self.root.destroy()

    def _update_progress(self, _value: int) -> None:
        self.value_label.config(text=f"Progress: {_value} from {self.max_value}")
        self.progressbar["value"] = _value
