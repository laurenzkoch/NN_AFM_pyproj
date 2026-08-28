import tkinter as tk
from tkinter import ttk
import random


class IndicatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Indikator-Oberfläche")
        self.root.geometry("450x300")

        self.running = False
        self.value = 0

        # Überschrift
        title_label = ttk.Label(
            root,
            text="Systemüberwachung",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=15)

        # Statusanzeige
        status_frame = ttk.Frame(root)
        status_frame.pack(pady=10)

        ttk.Label(
            status_frame,
            text="Systemstatus:"
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

        self.status_label = ttk.Label(
            status_frame,
            text="Gestoppt"
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

        # Messwert
        self.value_label = ttk.Label(
            root,
            text="Messwert: 0 %",
            font=("Arial", 14)
        )
        self.value_label.pack(pady=10)

        # Fortschrittsanzeige
        self.progressbar = ttk.Progressbar(
            root,
            orient="horizontal",
            length=350,
            mode="determinate",
            maximum=100
        )
        self.progressbar.pack(pady=10)

        # Start-/Stopp-Schaltfläche
        self.start_button = ttk.Button(
            root,
            text="Start",
            command=self.toggle_simulation
        )
        self.start_button.pack(pady=15)

    def toggle_simulation(self) -> None:
        self.running = not self.running

        if self.running:
            self.start_button.config(text="Stopp")
            self.update_indicators()
        else:
            self.start_button.config(text="Start")
            self.set_status("Gestoppt", "red")

    def update_indicators(self) -> None:
        if not self.running:
            return

        # Beispielwert simulieren
        self.value = random.randint(0, 100)

        self.value_label.config(
            text=f"Messwert: {self.value} %"
        )
        self.progressbar["value"] = self.value

        # Indikatorfarbe abhängig vom Messwert
        if self.value < 60:
            self.set_status("Normal", "green")
        elif self.value < 85:
            self.set_status("Warnung", "orange")
        else:
            self.set_status("Kritisch", "red")

        # Nach 500 ms erneut ausführen
        self.root.after(500, self.update_indicators)

    def set_status(self, text: str, color: str) -> None:
        self.status_indicator.itemconfig(
            self.status_led,
            fill=color
        )
        self.status_label.config(text=text)


def main() -> None:
    root = tk.Tk()
    IndicatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()