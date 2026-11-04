import csv
import sys
from datetime import datetime
from pathlib import Path
from statistics import fmean


DATETIME_FORMAT = "%Y-%m-%d %H:%M"
PERIODS = (
    ("1주차 (2025-10-01~07)", "weather.csv", "2025-10-01", "2025-10-07"),
    ("2주차 (2025-10-08~14)", "weather_week2.csv", "2025-10-08", "2025-10-14"),
)
MEASUREMENTS = (
    ("temperature_C", "기온", "°C"),
    ("humidity_pct", "습도", "%"),
    ("wind_speed_ms", "풍속", "m/s"),
)


def load_period(
    csv_path: Path, expected_start: str, expected_end: str
) -> tuple[dict[str, list[float]], int]:
    """Load one period, validate timestamps, and collect available measurements."""
    measurements = {column: [] for column, _, _ in MEASUREMENTS}
    timestamps = []
    required_columns = {"datetime", *(column for column, _, _ in MEASUREMENTS)}

    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"{csv_path.name}에 필요한 열이 없습니다: "
                + ", ".join(sorted(required_columns))
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.strptime(row["datetime"], DATETIME_FORMAT)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{csv_path.name} {line_number}행의 datetime 형식이 올바르지 않습니다."
                ) from exc
            timestamps.append(timestamp)

            for column, _, _ in MEASUREMENTS:
                raw_value = (row[column] or "").strip()
                if not raw_value:
                    continue
                try:
                    measurements[column].append(float(raw_value))
                except ValueError as exc:
                    raise ValueError(
                        f"{csv_path.name} {line_number}행의 {column} 값이 숫자가 아닙니다."
                    ) from exc

    if not timestamps:
        raise ValueError(f"{csv_path.name}에 비교할 데이터가 없습니다.")
    if min(timestamps).date().isoformat() != expected_start:
        raise ValueError(
            f"{csv_path.name}의 첫 날짜가 {expected_start}이 아닙니다."
        )
    if max(timestamps).date().isoformat() != expected_end:
        raise ValueError(
            f"{csv_path.name}의 마지막 날짜가 {expected_end}이 아닙니다."
        )
    if len(set(timestamps)) != len(timestamps):
        raise ValueError(f"{csv_path.name}에 중복 시각이 있습니다.")

    return measurements, len(timestamps)


def summarize(values: list[float]) -> tuple[float, float, float] | None:
    """Return mean, minimum, and maximum, or None when a field is all missing."""
    if not values:
        return None
    return fmean(values), min(values), max(values)


def main() -> int:
    """Compare weekly mean, minimum, maximum, and missing counts."""
    folder = Path(__file__).resolve().parent
    summaries = []

    try:
        for _, filename, expected_start, expected_end in PERIODS:
            data, row_count = load_period(
                folder / filename, expected_start, expected_end
            )
            summaries.append(
                {
                    "row_count": row_count,
                    "measurements": {
                        column: summarize(data[column])
                        for column, _, _ in MEASUREMENTS
                    },
                    "valid_counts": {
                        column: len(data[column])
                        for column, _, _ in MEASUREMENTS
                    },
                }
            )
    except (OSError, csv.Error, ValueError) as exc:
        print(f"비교 실패: {exc}")
        return 1

    print("주간 기상자료 비교 (2주차 - 1주차)")
    print(f"{'항목':<12} {'통계':<8} {'1주차':>12} {'2주차':>12} {'차이':>12}")
    print("-" * 60)

    for column, label, unit in MEASUREMENTS:
        first = summaries[0]["measurements"][column]
        second = summaries[1]["measurements"][column]
        print(f"{label} ({unit})")
        for index, statistic in enumerate(("평균", "최솟값", "최댓값")):
            first_value = first[index] if first is not None else None
            second_value = second[index] if second is not None else None
            difference = (
                second_value - first_value
                if first_value is not None and second_value is not None
                else None
            )
            formatted = [
                f"{value:.2f}" if value is not None else "자료 없음"
                for value in (first_value, second_value, difference)
            ]
            print(
                f"{'':<12} {statistic:<8} "
                f"{formatted[0]:>12} {formatted[1]:>12} {formatted[2]:>12}"
            )

        for period_index, period in enumerate(PERIODS):
            valid_count = summaries[period_index]["valid_counts"][column]
            row_count = summaries[period_index]["row_count"]
            missing_count = 168 - valid_count
            print(
                f"  {period[0]} 유효 {valid_count}개 / 결측 {missing_count}개 "
                f"(CSV 행 {row_count}개)"
            )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
