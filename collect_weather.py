import csv
import getpass
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, quote_plus

import requests


API_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
FIELD_POSITIONS = {"TM": 1, "WS": 4, "TA": 12, "HM": 14}
OUTPUT_COLUMNS = [
    "datetime",
    "temperature_C",
    "humidity_pct",
    "wind_speed_ms",
]
MISSING_VALUES = {
    "TM": {"", "-9", "-99", "-999", "-9999"},
    "WS": {"", "-9", "-9.0", "-99", "-99.0", "-99.9", "-999", "-999.0"},
    "TA": {"", "-99", "-99.0", "-99.9", "-999", "-999.0", "-9999"},
    "HM": {"", "-9", "-9.0", "-99", "-99.0", "-99.9", "-999", "-999.0"},
}


def _read_response_body(response: requests.Response) -> str:
    # Decode using the response charset, falling back to detected encoding.
    encoding = requests.utils.get_encoding_from_headers(response.headers)
    if encoding is None:
        encoding = response.apparent_encoding
    response.encoding = encoding
    return response.text


def _message_from_body(body: str) -> str:
    # Extract an API error message when the endpoint returns a JSON error.
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict) and "message" in payload:
        return str(payload["message"])
    return ""


def _redact(text: str, api_key: str, request_url: str) -> str:
    # Prevent credentials or the credential-bearing URL from reaching console output.
    if api_key:
        for secret in {api_key, quote(api_key, safe=""), quote_plus(api_key)}:
            text = text.replace(secret, "[인증키 숨김]")
    if request_url:
        text = text.replace(request_url, "[요청 주소 숨김]")
    text = re.sub(
        r"(?i)(authkey=)[^&\s\"'<>]+",
        r"\1[인증키 숨김]",
        text,
    )
    return text


def _separate_response_lines(body: str) -> tuple[list[str], list[tuple[int, list[str]]]]:
    # Keep API explanation lines separate and split each data row on whitespace.
    explanations = []
    data_rows = []
    for line_number, line in enumerate(body.splitlines(), start=1):
        if line.startswith("#"):
            explanations.append(line)
        elif line.strip():
            data_rows.append((line_number, line.split()))
    return explanations, data_rows


def _missing_value(field: str, value: str) -> str:
    # Use field-specific sentinels so ordinary negative temperatures remain valid.
    normalized = value.strip()
    return "" if normalized in MISSING_VALUES[field] else normalized


def _make_weather_rows(
    data_rows: list[tuple[int, list[str]]],
) -> tuple[list[dict[str, str]], int]:
    # Convert the configured 1-based KMA field positions into the CSV schema.
    required_position = max(FIELD_POSITIONS.values())
    parsed_rows = []
    seen_by_time = {}
    duplicate_count = 0

    for line_number, values in data_rows:
        if len(values) < required_position:
            raise ValueError(
                f"응답 {line_number}번째 줄에 필드가 부족합니다 "
                f"(필요: {required_position}, 실제: {len(values)})."
            )

        raw_time = values[FIELD_POSITIONS["TM"] - 1]
        try:
            timestamp = datetime.strptime(raw_time, "%Y%m%d%H%M")
        except ValueError as exc:
            raise ValueError(
                f"응답 {line_number}번째 줄의 TM 형식이 올바르지 않습니다."
            ) from exc

        row = {
            "datetime": timestamp.strftime("%Y-%m-%d %H:%M"),
            "temperature_C": _missing_value(
                "TA", values[FIELD_POSITIONS["TA"] - 1]
            ),
            "humidity_pct": _missing_value(
                "HM", values[FIELD_POSITIONS["HM"] - 1]
            ),
            "wind_speed_ms": _missing_value(
                "WS", values[FIELD_POSITIONS["WS"] - 1]
            ),
        }

        previous = seen_by_time.get(timestamp)
        if previous is not None:
            if previous != row:
                raise ValueError(
                    f"{row['datetime']} 시각의 중복 자료 값이 서로 다릅니다."
                )
            duplicate_count += 1
            continue

        seen_by_time[timestamp] = row
        parsed_rows.append((timestamp, row))

    parsed_rows.sort(key=lambda item: item[0])
    return [row for _, row in parsed_rows], duplicate_count


def _save_weather_csv(
    rows: list[dict[str, str]], output_path: Path
) -> None:
    # Write only the requested columns, without a dataframe index.
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    api_key = getpass.getpass("KMA API 인증키: ")
    params = {
        "tm1": "202510080000",
        "tm2": "202510142300",
        "stn": "108",
        "help": "0",
        "authKey": api_key,
    }

    try:
        response = requests.get(API_URL, params=params, timeout=30)
    except requests.RequestException:
        print("요청 중 오류가 발생했습니다. 인증키와 요청 주소는 표시하지 않습니다.")
        return 1

    print(f"HTTP 상태 코드: {response.status_code}")
    output_directory = Path(__file__).resolve().parent
    raw_path = output_directory / "weather_raw_week2.txt"
    raw_path.write_bytes(response.content)
    body = _read_response_body(response)
    message = _message_from_body(body)

    if not response.ok or message:
        safe_message = _redact(
            message or body.strip() or "응답 본문에 오류 메시지가 없습니다.",
            api_key,
            response.url,
        )
        print(f"요청 실패: {safe_message}")
        return 1

    try:
        explanation_lines, data_rows = _separate_response_lines(body)
        weather_rows, duplicate_count = _make_weather_rows(data_rows)
    except ValueError as exc:
        print(f"데이터 처리 실패: {exc}")
        return 1

    csv_path = output_directory / "weather_week2.csv"
    _save_weather_csv(weather_rows, csv_path)

    print(f"설명 줄 수: {len(explanation_lines)}")
    print(
        "사용한 필드 위치(1부터 시작): "
        "TM(1), WS(4), TA(12), HM(14)"
    )
    print(f"중복 자료 행 제거 수: {duplicate_count}")
    print(f"저장 행 수: {len(weather_rows)}")
    print(
        "열별 결측 수: "
        + ", ".join(
            f"{column}={sum(not row[column] for row in weather_rows)}"
            for column in OUTPUT_COLUMNS
        )
    )
    print(
        f"첫 시각: {weather_rows[0]['datetime'] if weather_rows else '(자료 없음)'}"
    )
    print(
        f"마지막 시각: {weather_rows[-1]['datetime'] if weather_rows else '(자료 없음)'}"
    )
    print(f"원본 응답 저장 위치: {raw_path}")
    print(f"CSV 저장 위치: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())