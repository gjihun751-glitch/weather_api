import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


DATETIME_FORMAT = "%Y-%m-%d %H:%M"


def load_hourly_temperatures(
    csv_path: Path,
) -> tuple[list[datetime], list[float | None], list[datetime]]:
    """Read, validate, and sort temperatures, inserting missing hourly times."""
    temperatures_by_time = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"datetime", "temperature_C"}
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "weather.csv에 datetime과 temperature_C 열이 필요합니다."
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.strptime(row["datetime"], DATETIME_FORMAT)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"CSV {line_number}행의 datetime 형식이 올바르지 않습니다."
                ) from exc

            if timestamp.minute != 0:
                raise ValueError(
                    f"CSV {line_number}행의 시각이 정각 단위가 아닙니다."
                )
            if timestamp in temperatures_by_time:
                raise ValueError(
                    f"CSV에 중복 시각이 있습니다: {timestamp.strftime(DATETIME_FORMAT)}"
                )

            raw_temperature = (row["temperature_C"] or "").strip()
            try:
                temperature = float(raw_temperature) if raw_temperature else None
            except ValueError as exc:
                raise ValueError(
                    f"CSV {line_number}행의 temperature_C가 숫자가 아닙니다."
                ) from exc
            temperatures_by_time[timestamp] = temperature

    if not temperatures_by_time:
        raise ValueError("weather.csv에 그릴 데이터가 없습니다.")

    first_time = min(temperatures_by_time)
    last_time = max(temperatures_by_time)
    hour_count = int((last_time - first_time).total_seconds() // 3600) + 1
    times = [first_time + timedelta(hours=offset) for offset in range(hour_count)]
    missing_times = [
        timestamp for timestamp in times if timestamp not in temperatures_by_time
    ]
    values = [temperatures_by_time.get(timestamp) for timestamp in times]
    return times, values, missing_times


def plot_temperature(
    times: list[datetime], temperatures: list[float | None], output_path: Path
) -> None:
    """Save and display the temperature line chart; None values break the line."""
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(times, temperatures, linewidth=1.5)
    axis.set_title("Temperature over time")
    axis.set_xlabel("Time (KST)")
    axis.set_ylabel("Temperature (°C)")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.show()
    plt.close(figure)


def main() -> int:
    """Load week 2 weather data and produce its temperature chart."""
    folder = Path(__file__).resolve().parent
    csv_path = folder / "weather_week2.csv"
    image_path = folder / "temperature_week2.png"

    try:
        times, temperatures, missing_times = load_hourly_temperatures(csv_path)
        plot_temperature(times, temperatures, image_path)
    except (OSError, csv.Error, ValueError) as exc:
        print(f"그래프 생성 실패: {exc}")
        return 1

    print(f"총 시간 구간: {len(times)}")
    print(f"누락 시각 수: {len(missing_times)}")
    if missing_times:
        print(
            "누락 시각: "
            + ", ".join(timestamp.strftime(DATETIME_FORMAT) for timestamp in missing_times)
        )
    print(f"기온 결측 구간 수: {sum(value is None for value in temperatures)}")
    print(f"이미지 저장 위치: {image_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
