$ErrorActionPreference = 'Stop'

$sourcePath = Join-Path $PSScriptRoot 'asos_108_2023_2024.csv'
$outputPath = Join-Path $PSScriptRoot 'wind_dataset.csv'
$culture = [Globalization.CultureInfo]::InvariantCulture
$format = 'yyyy-MM-dd HH:mm:ss'

$sourceRows = @(Import-Csv -LiteralPath $sourcePath -Encoding UTF8)
$observations = @{}
$timestamps = [System.Collections.Generic.List[datetime]]::new()

foreach ($row in $sourceRows) {
    $timestamp = [datetime]::MinValue
    if (-not [datetime]::TryParseExact(
        $row.datetime,
        $format,
        $culture,
        [Globalization.DateTimeStyles]::None,
        [ref]$timestamp
    )) {
        throw "Invalid timestamp: $($row.datetime)"
    }

    $key = $timestamp.ToString($format, $culture)
    if ($observations.ContainsKey($key)) {
        throw "Duplicate timestamp: $key"
    }

    $observations[$key] = $row
    $timestamps.Add($timestamp)
}

if ($timestamps.Count -eq 0) {
    throw 'The source CSV contains no observations.'
}

$start = ($timestamps | Measure-Object -Minimum).Minimum
$end = ($timestamps | Measure-Object -Maximum).Maximum
$hourlyRows = [System.Collections.Generic.List[object]]::new()

for ($timestamp = $start; $timestamp -le $end; $timestamp = $timestamp.AddHours(1)) {
    $key = $timestamp.ToString($format, $culture)
    if ($observations.ContainsKey($key)) {
        $hourlyRows.Add($observations[$key])
    }
    else {
        $hourlyRows.Add($null)
    }
}

$dataset = [System.Collections.Generic.List[object]]::new()
for ($index = 2; $index -lt ($hourlyRows.Count - 1); $index++) {
    $current = $hourlyRows[$index]
    $oneHourBefore = $hourlyRows[$index - 1]
    $twoHoursBefore = $hourlyRows[$index - 2]
    $next = $hourlyRows[$index + 1]

    if ($null -eq $current -or $null -eq $oneHourBefore -or
        $null -eq $twoHoursBefore -or $null -eq $next) {
        continue
    }

    $requiredValues = @(
        $current.wind_speed_ms,
        $current.temperature_C,
        $current.humidity_pct,
        $oneHourBefore.wind_speed_ms,
        $twoHoursBefore.wind_speed_ms,
        $next.wind_speed_ms
    )
    if (@($requiredValues | Where-Object { [string]::IsNullOrWhiteSpace($_) }).Count -gt 0) {
        continue
    }

    $predictionTime = $start.AddHours($index)
    $targetTime = $predictionTime.AddHours(1)
    $dataset.Add([pscustomobject][ordered]@{
        prediction_time       = $predictionTime.ToString($format, $culture)
        target_time           = $targetTime.ToString($format, $culture)
        wind_speed_ms_current = $current.wind_speed_ms
        temperature_C_current = $current.temperature_C
        humidity_pct_current  = $current.humidity_pct
        wind_speed_ms_1h_before = $oneHourBefore.wind_speed_ms
        wind_speed_ms_2h_before = $twoHoursBefore.wind_speed_ms
        actual_wind_speed_ms  = $next.wind_speed_ms
    })
}

$dataset | Export-Csv -LiteralPath $outputPath -NoTypeInformation -Encoding UTF8

Write-Output "Hourly rows after gap alignment: $($hourlyRows.Count)"
Write-Output "Saved dataset rows: $($dataset.Count)"
Write-Output 'First 4 rows:'
$dataset | Select-Object -First 4 | Format-Table -AutoSize
