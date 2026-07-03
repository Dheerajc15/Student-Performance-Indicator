"""Evaluate the deployed model's accuracy and plot predicted vs actual scores.

Regression 'accuracy' = how close predictions are to actual math scores.
Loads the SAME artifacts the web app serves, so these numbers reflect
exactly what production returns.

Outputs:
  - Metrics printed to stdout (R2 / MAE / RMSE for train and test)
  - images/pred_vs_actual.png  (predicted vs actual on the test set)

Requires the dev/notebook environment (matplotlib): pip install -r requirements-dev.txt
"""
import os

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.utils import load_object

TARGET = "math_score"
IMAGE_DIR = "images"
IMAGE_PATH = os.path.join(IMAGE_DIR, "pred_vs_actual.png")

preprocessor = load_object("artifacts/preprocessor.pkl")
model = load_object("artifacts/model.pkl")


def predict(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    y_pred = model.predict(preprocessor.transform(X))
    return y, y_pred


def report(name, y, y_pred):
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred) ** 0.5
    print(f"{name:6s} (n={len(y):4d})  R2={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}")
    return r2, mae, rmse


def plot(y_test, y_pred_test, r2, rmse):
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    os.makedirs(IMAGE_DIR, exist_ok=True)
    lo, hi = 0, 100

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred_test, alpha=0.6, edgecolor="white", s=45,
               color="#2563eb", label="Test predictions")
    ax.plot([lo, hi], [lo, hi], "--", color="#dc2626", linewidth=1.5,
            label="Perfect prediction (y = x)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual math score")
    ax.set_ylabel("Predicted math score")
    ax.set_title("Predicted vs Actual Math Scores (Test Set)")
    ax.text(0.05, 0.92, f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(IMAGE_PATH, dpi=120)
    print(f"\nSaved plot -> {IMAGE_PATH}")


if __name__ == "__main__":
    print(f"Model: {type(model).__name__}\n")

    y_train, yp_train = predict("artifacts/train.csv")
    report("train", y_train, yp_train)

    y_test, yp_test = predict("artifacts/test.csv")
    r2, mae, rmse = report("test", y_test, yp_test)

    plot(y_test, yp_test, r2, rmse)
