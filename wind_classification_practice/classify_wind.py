import csv
from datetime import datetime, timedelta
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "wind_dataset.csv"
OUTPUT_PATH = BASE_DIR / "classification_predictions.csv"
SCORES_PATH = BASE_DIR / "classification_scores.csv"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
ONE_HOUR = timedelta(hours=1)
FEATURE_COLUMNS = (
    "wind_speed_ms_current",
    "temperature_C_current",
    "humidity_pct_current",
    "wind_speed_ms_1h_before",
    "wind_speed_ms_2h_before",
)
PREDICTION_COLUMNS = (
    ("Always 0", "always_0_prediction"),
    ("Current wind >= 4.0 m/s", "current_wind_threshold_prediction"),
    ("Logistic regression (probability >= 0.50)", "logistic_prediction"),
    ("Logistic regression (probability >= 0.20)", "logistic_prediction_0_20"),
)
LOGISTIC_THRESHOLDS = {
    "logistic_prediction": 0.50,
    "logistic_prediction_0_20": 0.20,
}


def parse_timestamp(value: str, column: str, line_number: int) -> datetime:
    try:
        return datetime.strptime(value, TIMESTAMP_FORMAT)
    except ValueError as error:
        raise ValueError(
            f"Invalid {column} on CSV line {line_number}: {value!r}"
        ) from error


def calculate_scores(
    actual_labels: list[int], predicted_labels: list[int]
) -> dict[str, int | float | str]:
    hits = sum(actual == 1 and predicted == 1 for actual, predicted in zip(actual_labels, predicted_labels))
    misses = sum(actual == 1 and predicted == 0 for actual, predicted in zip(actual_labels, predicted_labels))
    false_alarms = sum(actual == 0 and predicted == 1 for actual, predicted in zip(actual_labels, predicted_labels))
    correct_rejections = sum(
        actual == 0 and predicted == 0
        for actual, predicted in zip(actual_labels, predicted_labels)
    )
    predicted_positive_count = hits + false_alarms
    actual_positive_count = hits + misses
    total_count = len(actual_labels)

    return {
        "evaluation_rows": total_count,
        "actual_strong_wind_count": actual_positive_count,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_rejections": correct_rejections,
        "accuracy": (hits + correct_rejections) / total_count,
        "precision": hits / predicted_positive_count if predicted_positive_count else "N/A",
        "recall": hits / actual_positive_count if actual_positive_count else "N/A",
    }


def main() -> None:
    required_columns = {
        "prediction_time",
        "target_time",
        "actual_wind_speed_ms",
        *FEATURE_COLUMNS,
    }

    with INPUT_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

        rows = []
        previous_prediction_time = None
        previous_target_time = None

        for line_number, row in enumerate(reader, start=2):
            prediction_time = parse_timestamp(
                row["prediction_time"], "prediction_time", line_number
            )
            target_time = parse_timestamp(row["target_time"], "target_time", line_number)
            if target_time - prediction_time != ONE_HOUR:
                raise ValueError(
                    f"CSV line {line_number} is not exactly one hour apart: "
                    f"{row['prediction_time']} -> {row['target_time']}"
                )
            if (
                previous_prediction_time is not None
                and prediction_time < previous_prediction_time
            ):
                raise ValueError(
                    f"prediction_time is not chronological on CSV line {line_number}."
                )
            if previous_target_time is not None and target_time < previous_target_time:
                raise ValueError(
                    f"target_time is not chronological on CSV line {line_number}."
                )

            try:
                next_hour_wind_speed = float(row["actual_wind_speed_ms"])
                features = [float(row[column]) for column in FEATURE_COLUMNS]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric input on CSV line {line_number}."
                ) from error

            rows.append(
                {
                    "prediction_time": row["prediction_time"],
                    "target_time": row["target_time"],
                    "actual_wind_speed_ms": row["actual_wind_speed_ms"],
                    "label": int(next_hour_wind_speed >= 4.0),
                    "_actual_wind_speed_ms": next_hour_wind_speed,
                    "_features": features,
                    "_target_time": target_time,
                }
            )
            previous_prediction_time = prediction_time
            previous_target_time = target_time

    train_start = datetime(2023, 1, 1)
    evaluation_start = datetime(2024, 12, 1)
    evaluation_end = datetime(2025, 1, 1)
    training_rows = [
        row for row in rows if train_start <= row["_target_time"] < evaluation_start
    ]
    evaluation_rows = [
        row for row in rows if evaluation_start <= row["_target_time"] < evaluation_end
    ]

    selected_count = len(training_rows) + len(evaluation_rows)
    if selected_count != len(rows):
        raise ValueError("Dataset contains rows outside the requested date ranges.")
    if not training_rows or not evaluation_rows:
        raise ValueError("Training or December 2024 evaluation data is empty.")
    if {row["label"] for row in training_rows} != {0, 1}:
        raise ValueError("Training data must contain both wind-speed classes.")

    scaler = StandardScaler()
    training_features = [row["_features"] for row in training_rows]
    evaluation_features = [row["_features"] for row in evaluation_rows]
    scaled_training_features = scaler.fit_transform(training_features)
    scaled_evaluation_features = scaler.transform(evaluation_features)

    classifier = LogisticRegression()
    classifier.fit(scaled_training_features, [row["label"] for row in training_rows])
    strong_wind_class_index = list(classifier.classes_).index(1)
    strong_wind_probabilities = classifier.predict_proba(scaled_evaluation_features)[
        :, strong_wind_class_index
    ]

    def print_counts(name: str, selected_rows: list[dict[str, object]]) -> None:
        zero_count = sum(row["label"] == 0 for row in selected_rows)
        one_count = sum(row["label"] == 1 for row in selected_rows)
        print(f"{name}:")
        print(f"  Total rows: {len(selected_rows)}")
        print(f"  Label 0 (< 4.0 m/s): {zero_count}")
        print(f"  Label 1 (>= 4.0 m/s): {one_count}")

    print(f"Dataset: {INPUT_PATH.name}")
    print("Timestamps: KST")
    print(f"Chronological order: verified for all {len(rows)} rows")
    print(f"Prediction-to-target interval: exactly 1 hour for all {len(rows)} rows")
    print_counts("Training (target_time: 2023-01 through 2024-11)", training_rows)
    print_counts("Evaluation (target_time: 2024-12-01 through 2024-12-31)", evaluation_rows)

    first_evaluation_row = evaluation_rows[0]
    print("First evaluation row:")
    print(
        "  "
        f"prediction_time={first_evaluation_row['prediction_time']}, "
        f"target_time={first_evaluation_row['target_time']}, "
        f"actual_wind_speed_ms={first_evaluation_row['actual_wind_speed_ms']}, "
        f"label={first_evaluation_row['label']}"
    )

    fieldnames = (
        "prediction_time",
        "target_time",
        "actual_wind_speed_ms",
        "actual_label",
        "current_wind_speed_ms",
        "always_0_prediction",
        "current_wind_threshold_prediction",
        "logistic_strong_wind_probability",
        "logistic_prediction",
        "logistic_prediction_0_20",
    )
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row, strong_wind_probability in zip(
            evaluation_rows, strong_wind_probabilities
        ):
            current_wind_speed = row["_features"][0]
            writer.writerow(
                {
                    "prediction_time": row["prediction_time"],
                    "target_time": row["target_time"],
                    "actual_wind_speed_ms": row["actual_wind_speed_ms"],
                    "actual_label": row["label"],
                    "current_wind_speed_ms": current_wind_speed,
                    "always_0_prediction": 0,
                    "current_wind_threshold_prediction": int(
                        current_wind_speed >= 4.0
                    ),
                    "logistic_strong_wind_probability": strong_wind_probability,
                    "logistic_prediction": int(strong_wind_probability >= 0.50),
                    "logistic_prediction_0_20": int(
                        strong_wind_probability >= 0.20
                    ),
                }
            )

    actual_labels = [row["label"] for row in evaluation_rows]
    score_rows = []
    for method_name, prediction_column in PREDICTION_COLUMNS:
        predicted_labels = []
        for row, strong_wind_probability in zip(
            evaluation_rows, strong_wind_probabilities
        ):
            current_wind_speed = row["_features"][0]
            if prediction_column == "always_0_prediction":
                predicted_labels.append(0)
            elif prediction_column == "current_wind_threshold_prediction":
                predicted_labels.append(int(current_wind_speed >= 4.0))
            else:
                predicted_labels.append(
                    int(
                        strong_wind_probability
                        >= LOGISTIC_THRESHOLDS[prediction_column]
                    )
                )
        if len(predicted_labels) != len(actual_labels):
            raise ValueError(f"{method_name} predictions do not match evaluation rows.")
        score_rows.append(
            {"method": method_name, **calculate_scores(actual_labels, predicted_labels)}
        )

    score_fieldnames = (
        "method",
        "evaluation_rows",
        "actual_strong_wind_count",
        "hits",
        "misses",
        "false_alarms",
        "correct_rejections",
        "accuracy",
        "precision",
        "recall",
    )
    with SCORES_PATH.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=score_fieldnames)
        writer.writeheader()
        writer.writerows(score_rows)

    print(
        f"Compared the same {len(evaluation_rows)} December evaluation rows "
        f"for all methods; actual strong-wind rows: "
        f"{sum(actual_labels)}."
    )
    print(f"Scores saved to: {SCORES_PATH.name}")
    print(
        f"{'Method':<44} {'Hits':>5} {'Misses':>6} {'False alarms':>13} "
        f"{'Correct neg.':>13} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}"
    )
    for score in score_rows:
        precision_display = (
            score["precision"]
            if isinstance(score["precision"], str)
            else f"{score['precision']:.4f}"
        )
        recall_display = (
            score["recall"]
            if isinstance(score["recall"], str)
            else f"{score['recall']:.4f}"
        )
        print(
            f"{score['method']:<44} {score['hits']:>5} {score['misses']:>6} "
            f"{score['false_alarms']:>13} {score['correct_rejections']:>13} "
            f"{score['accuracy']:>10.4f} {precision_display:>10} "
            f"{recall_display:>10}"
        )

    print(f"Predictions saved to: {OUTPUT_PATH.name} ({len(evaluation_rows)} rows)")
    print("First 5 prediction CSV rows:")
    with OUTPUT_PATH.open("r", encoding="utf-8-sig", newline="") as predictions_file:
        for _ in range(6):
            csv_line = predictions_file.readline()
            if not csv_line:
                break
            print(csv_line.rstrip())


if __name__ == "__main__":
    main()
