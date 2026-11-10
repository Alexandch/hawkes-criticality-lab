param(
    [string]$Output = "data/raw/nypd_complaints_2024_2025_verified.csv"
)

$ErrorActionPreference = "Stop"
$baseUrl = "https://data.cityofnewyork.us/resource/qgea-i56i.json"
$columns = "cmplnt_num,cmplnt_fr_dt,cmplnt_fr_tm,ofns_desc"
$parent = Split-Path -Parent $Output
if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
if ((Test-Path -LiteralPath $Output) -and (Get-Item -LiteralPath $Output).Length -gt 0) {
    throw "Output already exists; choose another path or move the existing file first: $Output"
}

$downloaded = 0
$expected = 0
foreach ($year in 2024..2025) {
    foreach ($month in 1..12) {
        $start = [datetime]::new($year, $month, 1).ToString("yyyy-MM-dd")
        $end = [datetime]::new($year, $month, 1).AddMonths(1).ToString("yyyy-MM-dd")
        $where = "cmplnt_fr_dt >= '$start' AND cmplnt_fr_dt < '$end'"
        $encodedWhere = [uri]::EscapeDataString($where)
        $countUri = "${baseUrl}?`$select=count(*)&`$where=$encodedWhere"
        $monthCount = Invoke-RestMethod -Uri $countUri -TimeoutSec 120
        $monthExpected = [int]$monthCount[0].count
        $expected += $monthExpected
        $uri = "${baseUrl}?`$select=$columns&`$where=$encodedWhere&`$limit=100000"
        $rows = Invoke-RestMethod -Uri $uri -TimeoutSec 180
        if ($rows.Count -ne $monthExpected) {
            throw "Month ${start}: expected $monthExpected rows but received $($rows.Count)"
        }

        $csv = @($rows | Select-Object cmplnt_num, cmplnt_fr_dt, cmplnt_fr_tm, ofns_desc | ConvertTo-Csv -NoTypeInformation)
        if ($downloaded -eq 0) {
            $csv | Set-Content -LiteralPath $Output -Encoding utf8
        } else {
            $csv | Select-Object -Skip 1 | Add-Content -LiteralPath $Output -Encoding utf8
        }
        $downloaded += $rows.Count
        Write-Host "$start downloaded: $($rows.Count); total $downloaded"
    }
}

if ($downloaded -ne $expected) {
    throw "Expected $expected rows but downloaded $downloaded"
}
Write-Host "Saved $downloaded rows to $((Resolve-Path $Output).Path)"
