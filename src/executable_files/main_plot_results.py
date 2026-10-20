from src.modules.plotting.prediction_report import create_prediction_report
from src.modules.plotting.prediction_results import plot_prediction_results

MODEL_FOLDER: str = "20261020_161109_f1353af2"

if __name__ == "__main__":
    plot_prediction_results(MODEL_FOLDER)
    create_prediction_report(MODEL_FOLDER)
