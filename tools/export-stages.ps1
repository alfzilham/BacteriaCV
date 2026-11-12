# Ekspor hasil demo untuk halaman /app dan landing.
# Memanggil API lokal untuk 4 gambar contoh, lalu menyimpan JSON hasil dan PNG kelima tahap
# ke apps\site\public\demo. Jalankan SEKALI, dengan API aktif di jendela lain:
#   cd apps\api
#   uvicorn app.main:app --port 8000
# Pakai dari akar repo:  powershell -File tools\export-stages.ps1
param(
  [string]$ApiUrl = "http://localhost:8000"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$samples = Join-Path $root "apps\api\samples"
$out = Join-Path $root "apps\site\public\demo"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$health = Invoke-RestMethod -Uri "$ApiUrl/api/health" -TimeoutSec 15
if ($health.status -ne "ready") { throw "API belum siap: $($health.status). Tunggu model selesai dimuat." }

$files = @(Get-ChildItem -Path $samples -Filter "contoh-*.png")
if ($files.Count -ne 4) { throw "Diharapkan 4 gambar contoh di $samples, ditemukan $($files.Count)." }

$utf8 = New-Object System.Text.UTF8Encoding($false)
foreach ($img in $files) {
  $id = [System.IO.Path]::GetFileNameWithoutExtension($img.Name)
  $tmp = Join-Path $env:TEMP ("bcv-" + $id + ".json")
  & curl.exe -s -S -f -X POST -F ("file=@" + $img.FullName) ($ApiUrl + "/api/predict?stage=0") -o $tmp
  if ($LASTEXITCODE -ne 0) { throw "Panggilan API gagal untuk $($img.Name)." }

  $json = Get-Content -Path $tmp -Raw -Encoding UTF8 | ConvertFrom-Json
  $panels = @($json.panels)
  if ($panels.Count -ne 5) { throw "Diharapkan 5 panel untuk $id, diterima $($panels.Count)." }

  for ($i = 0; $i -lt 5; $i++) {
    $png = Join-Path $out ("{0}-stage-{1}.png" -f $id, ($i + 1))
    [System.IO.File]::WriteAllBytes($png, [System.Convert]::FromBase64String($panels[$i]))
  }

  # Panel disimpan sebagai PNG terpisah, jadi dikosongkan dari JSON.
  $json.panels = @()
  $json.panel = ""
  [System.IO.File]::WriteAllText((Join-Path $out ($id + ".json")), ($json | ConvertTo-Json -Depth 8), $utf8)
  Remove-Item -Path $tmp -Force
  Write-Output ("OK  " + $id)
}

Write-Output ("Selesai. Berkas ada di " + $out)
Get-ChildItem -Path $out | Select-Object Name, Length
