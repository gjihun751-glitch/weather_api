import csv
import math
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "wind_dataset.csv"
OUTPUT_PATH = BASE_DIR / "predictions.csv"
FEATURE_COLUMNS = (
    "wind_speed_ms_current",
    "wind_speed_ms_1h_before",
    "wind_speed_ms_2h_before",
    "temperature_C_current",
    "humidity_pct_current",
)
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_FORMAT)


def fit_linear_regression(
    features: list[list[float]], targets: list[float]
) -> tuple[list[float], list[float], list[float]]:
    means = [
        sum(row[column] for row in features) / len(features)
        for column in range(len(FEATURE_COLUMNS))
    ]
    scales = []
    for column, mean in enumerate(means):
        variance = sum((row[column] - mean) ** 2 for row in features) / len(features)
        scale = math.sqrt(variance)
        scales.append(scale if scale else 1.0)

    matrix = [
        [1.0] + [(value - means[i]) / scales[i] for i, value in enumerate(row)]
        for row in features
    ]
    transformed_targets = targets.copy()
    row_count = len(matrix)
    column_count = len(FEATURE_COLUMNS) + 1

    for column in range(column_count):
        norm = math.sqrt(
            sum(matrix[row][column] ** 2 for row in range(column, row_count))
        )
        if norm == 0:
            raise ValueError("Training features do not have full column rank.")

        first = matrix[column][column]
        alpha = -math.copysign(norm, first)
        reflector = [matrix[row][column] for row in range(column, row_count)]
        reflector[0] -= alpha
        denominator = sum(value * value for value in reflector)
        if denominator == 0:
            raise ValueError("Training features do not have full column rank.")
        factor = 2.0 / denominator

        for target_column in range(column, column_count):
            projection = factor * sum(
                reflector[offset] * matrix[column + offset][target_column]
                for offset in range(len(reflector))
            )
            for offset, value in enumerate(reflector):
                matrix[column + offset][target_column] -= projection * value

        projection = factor * sum(
            reflector[offset] * transformed_targets[column + offset]
            for offset in range(len(reflector))
        )
        for offset, value in enumerate(reflector):
            transformed_targets[column + offset] -= projection * value

    coefficients = [0.0] * column_count
    for row in range(column_count - 1, -1, -1):
        remainder = sum(
            matrix[row][column] * coefficients[column]
            for column in range(row + 1, column_count)
        )
        diagonal = matrix[row][row]
        if abs(diagonal) < 1e-12:
            raise ValueError("Training features do not have full column rank.")
        coefficients[row] = (transformed_targets[row] - remainder) / diagonal

    return coefficients, means, scales


def predict(features: list[float], model: tuple[list[float], list[float], list[float]]) -> float:
    coefficients, means, scales = model
    prediction = coefficients[0] + sum(
        coefficients[i + 1] * (value - means[i]) / scales[i]
        for i, value in enumerate(features)
    )
    return max(0.0, prediction)


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required_columns = {
            "prediction_time",
            "target_time",
            "actual_wind_speed_ms",
            *FEATURE_COLUMNS,
        }
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

        rows = []
        for row in reader:
            rows.append(
                {
                    **row,
                    "_prediction_dt": parse_timestamp(row["prediction_time"]),
                    "_target_dt": parse_timestamp(row["target_time"]),
                    "_features": [float(row[column]) for column in FEATURE_COLUMNS],
                    "_actual": float(row["actual_wind_speed_ms"]),
                }
            )

    rows.sort(key=lambda row: row["_target_dt"])
    train_start = datetime(2023, 1, 1)
    test_start = datetime(2024, 12, 1)
    test_end = datetime(2025, 1, 1)
    training_rows = [
        row for row in rows if train_start <= row["_target_dt"] < test_start
    ]
    evaluation_rows = [
        row for row in rows if test_start <= row["_target_dt"] < test_end
    ]
    if not training_rows or not evaluation_rows:
        raise ValueError("Training or December 2024 evaluation data is empty.")

    model = fit_linear_regression(
        [row["_features"] for row in training_rows],
        [row["_actual"] for row in training_rows],
    )
    forest = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
    )
    forest.fit(
        [row["_features"] for row in training_rows],
        [row["_actual"] for row in training_rows],
    )

    predictions = []
    for row in evaluation_rows:
        regression_prediction = predict(row["_features"], model)
        forest_prediction = max(
            0.0, float(forest.predict([row["_features"]])[0])
        )
        predictions.append(
            {
                "prediction_time": row["prediction_time"],
                "target_time": row["target_time"],
                "actual_wind_speed_ms": row["_actual"],
                "current_wind_speed_prediction_ms": row["_features"][0],
                "linear_regression_prediction_ms": regression_prediction,
                "random_forest_prediction_ms": forest_prediction,
            }
        )

    baseline_mae = sum(
        abs(row["actual_wind_speed_ms"] - row["current_wind_speed_prediction_ms"])
        for row in predictions
    ) / len(predictions)
    regression_mae = sum(
        abs(row["actual_wind_speed_ms"] - row["linear_regression_prediction_ms"])
        for row in predictions
    ) / len(predictions)
    forest_mae = sum(
        abs(row["actual_wind_speed_ms"] - row["random_forest_prediction_ms"])
        for row in predictions
    ) / len(predictions)

    fieldnames = (
        "prediction_time",
        "target_time",
        "actual_wind_speed_ms",
        "current_wind_speed_prediction_ms",
        "linear_regression_prediction_ms",
        "random_forest_prediction_ms",
    )
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Training rows: {len(training_rows)}")
    print(f"Evaluation rows: {len(evaluation_rows)}")
    print(f"Current-wind MAE: {baseline_mae:.6f} m/s")
    print(f"Linear-regression MAE: {regression_mae:.6f} m/s")
    print(f"Random-forest MAE: {forest_mae:.6f} m/s")
    print(
        "Random forest MAE change vs current wind: "
        f"{forest_mae - baseline_mae:+.6f} m/s "
        f"({(forest_mae / baseline_mae - 1) * 100:+.2f}%)"
    )
    print(
        "Random forest MAE change vs linear regression: "
        f"{forest_mae - regression_mae:+.6f} m/s "
        f"({(forest_mae / regression_mae - 1) * 100:+.2f}%)"
    )
    print("First evaluation result:")
    print(predictions[0])


if __name__ == "__main__":
    main()
