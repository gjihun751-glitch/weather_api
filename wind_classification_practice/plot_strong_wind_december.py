import csv
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "classification_predictions.csv"
OUTPUT_PATH = BASE_DIR / "strong_wind_december.png"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def main() -> None:
    required_columns = {
        "target_time",
        "actual_wind_speed_ms",
        "actual_label",
        "logistic_prediction",
        "logistic_prediction_0_20",
    }

    with INPUT_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Predictions file is missing columns: {sorted(missing_columns)}"
            )

        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                target_time = datetime.strptime(
                    row["target_time"], TIMESTAMP_FORMAT
                )
                actual_wind_speed = float(row["actual_wind_speed_ms"])
                actual_label = int(row["actual_label"])
                prediction_0_50 = int(row["logistic_prediction"])
                prediction_0_20 = int(row["logistic_prediction_0_20"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid prediction data on CSV line {line_number}."
                ) from error

            if actual_label not in (0, 1) or prediction_0_50 not in (0, 1):
                raise ValueError(f"Invalid binary label on CSV line {line_number}.")
            if prediction_0_20 not in (0, 1):
                raise ValueError(f"Invalid 0.20 prediction on CSV line {line_number}.")

            rows.append(
                {
                    "target_time": target_time,
                    "actual_wind_speed": actual_wind_speed,
                    "actual_label": actual_label,
                    "prediction_0_50": prediction_0_50,
                    "prediction_0_20": prediction_0_20,
                }
            )

    week_start = datetime(2024, 12, 1)
    week_end = datetime(2024, 12, 8)
    week_rows = [
        row for row in rows if week_start <= row["target_time"] < week_end
    ]
    week_rows.sort(key=lambda row: row["target_time"])
    if not week_rows:
        raise ValueError("No December 1-7, 2024 evaluation rows were found.")

    times = [row["target_time"] for row in week_rows]
    figure, (wind_axis, classification_axis) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, height_ratios=(2, 1)
    )

    wind_axis.plot(
        times,
        [row["actual_wind_speed"] for row in week_rows],
        color="black",
        linewidth=1.5,
        label="Actual wind speed",
    )
    wind_axis.axhline(
        4.0,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="Strong-wind threshold (4.0 m/s)",
    )
    wind_axis.set_ylabel("Wind speed (m/s)")
    wind_axis.set_title("Strong-wind classification: December 1-7, 2024 (KST)")
    wind_axis.legend()
    wind_axis.grid(True, alpha=0.25)

    classification_series = (
        ("Actual strong wind", "actual_label", "black", "o", -0.16),
        ("Prediction (p >= 0.50)", "prediction_0_50", "tab:blue", "s", 0.0),
        ("Prediction (p >= 0.20)", "prediction_0_20", "tab:orange", "^", 0.16),
    )
    for label, column, color, marker, offset in classification_series:
        classification_axis.scatter(
            times,
            [row[column] + offset for row in week_rows],
            color=color,
            marker=marker,
            s=24,
            label=label,
        )

    classification_axis.set_yticks([0, 1], labels=["No strong wind", "Strong wind"])
    classification_axis.set_ylim(-0.4, 1.4)
    classification_axis.set_ylabel("Classification")
    classification_axis.set_xlabel("Target time (KST)")
    classification_axis.set_xlim(week_start, week_end)
    classification_axis.xaxis.set_major_locator(mdates.DayLocator())
    classification_axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    classification_axis.legend(ncol=3, loc="upper right")
    classification_axis.grid(True, axis="x", alpha=0.25)

    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=150)
    plt.close(figure)

    december_2_rows = [
        row for row in week_rows if row["target_time"].date().isoformat() == "2024-12-02"
    ]
    incorrect_cases = [
        (row, threshold)
        for row in december_2_rows
        for threshold, prediction_column in (
            ("0.50", "prediction_0_50"),
            ("0.20", "prediction_0_20"),
        )
        if row["actual_label"] != row[prediction_column]
    ]

    print(f"Plotted evaluation rows (Dec 1-7): {len(week_rows)}")
    print(f"Saved plot: {OUTPUT_PATH}")
    if incorrect_cases:
        row, threshold = incorrect_cases[0]
        print(
            "December 2 incorrect case: "
            f"target_time={row['target_time']:%Y-%m-%d %H:%M} KST, "
            f"actual_wind_speed={row['actual_wind_speed']:.1f} m/s "
            f"(actual label={row['actual_label']}), "
            f"p >= {threshold} prediction="
            f"{row[f'prediction_{threshold.replace('.', '_')}']}"
        )
    else:
        print("No incorrectly classified December 2 rows were found.")


if __name__ == "__main__":
    main()
