import csv
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "predictions.csv"
OUTPUT_PATH = BASE_DIR / "prediction_week.png"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required_columns = {
            "target_time",
            "actual_wind_speed_ms",
            "current_wind_speed_prediction_ms",
            "linear_regression_prediction_ms",
            "random_forest_prediction_ms",
        }
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Predictions file is missing columns: {sorted(missing_columns)}")

        rows = []
        for row in reader:
            rows.append(
                {
                    "target_time": datetime.strptime(
                        row["target_time"], TIMESTAMP_FORMAT
                    ),
                    "actual": float(row["actual_wind_speed_ms"]),
                    "baseline": float(row["current_wind_speed_prediction_ms"]),
                    "regression": float(row["linear_regression_prediction_ms"]),
                    "forest": float(row["random_forest_prediction_ms"]),
                }
            )

    rows.sort(key=lambda row: row["target_time"])
    month_rows = [
        row
        for row in rows
        if datetime(2024, 12, 1) <= row["target_time"] < datetime(2025, 1, 1)
    ]
    week_rows = [
        row
        for row in month_rows
        if datetime(2024, 12, 1) <= row["target_time"] < datetime(2024, 12, 8)
    ]
    if not month_rows or not week_rows:
        raise ValueError("December 2024 month or first-week evaluation rows are empty.")

    baseline_mae = sum(
        abs(row["actual"] - row["baseline"]) for row in month_rows
    ) / len(month_rows)
    regression_mae = sum(
        abs(row["actual"] - row["regression"]) for row in month_rows
    ) / len(month_rows)
    forest_mae = sum(
        abs(row["actual"] - row["forest"]) for row in month_rows
    ) / len(month_rows)

    times = [row["target_time"] for row in week_rows]
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(times, [row["actual"] for row in week_rows], color="black", label="Actual")
    axis.plot(
        times,
        [row["baseline"] for row in week_rows],
        color="gray",
        linestyle="--",
        label="Current wind speed baseline",
    )
    axis.plot(
        times,
        [row["regression"] for row in week_rows],
        color="blue",
        label="Linear regression",
    )
    axis.plot(
        times,
        [row["forest"] for row in week_rows],
        color="darkorange",
        label="Random forest",
    )
    axis.set_xlabel("Target time (KST)")
    axis.set_ylabel("Wind speed (m/s)")
    axis.set_title("Wind speed predictions: December 1-7, 2024")
    axis.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 12)))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=150)
    plt.close(figure)

    print(f"Plot rows (Dec 1-7): {len(week_rows)}")
    print(f"Monthly evaluation rows (Dec 1-31): {len(month_rows)}")
    print(f"Current-wind baseline MAE: {baseline_mae:.6f} m/s")
    print(f"Linear-regression MAE: {regression_mae:.6f} m/s")
    print(f"Random-forest MAE: {forest_mae:.6f} m/s")
    print(f"Saved plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
