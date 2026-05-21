import ansys.motorcad.core as pymotorcad


def main() -> None:
    # Startet eine sichtbare Motor-CAD-Instanz.
    mcad = pymotorcad.MotorCAD()

    # Optional: Modell laden (Pfad anpassen und Kommentar entfernen).
    model_path = r"D:\\MotorCAD\\test.mot"
    mcad.load_from_file(model_path)

    try:
        mcad.set_visible(True)
    except Exception:
        pass

    print("Motor-CAD wurde gestartet.")


    mcad.set_variable("Slot_Number", 6)

    input("Zum Beenden Enter drücken...")
    mcad.quit()


if __name__ == "__main__":
    main()