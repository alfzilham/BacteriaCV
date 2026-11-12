# check_env.ps1
# Memeriksa kesiapan lingkungan. Selalu memakai Python dari .venv bila tersedia.

$ErrorActionPreference = "Continue"
$ok = $true

$venvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"

Write-Host "Memeriksa virtual environment..."
if (Test-Path $venvPython) {
    Write-Host "  OK  .venv ditemukan"
    $py = $venvPython
} else {
    Write-Host "  GAGAL  .venv belum dibuat, jalankan setup_env.ps1"
    $ok = $false
    $py = "python"
}

Write-Host "Memeriksa Python yang dipakai..."
$ver = & $py --version 2>&1
if ($LASTEXITCODE -eq 0) { Write-Host "  OK  $ver ($py)" } else { Write-Host "  GAGAL  Python tidak dapat dijalankan"; $ok = $false }

Write-Host "Memeriksa paket..."
$pkgs = @("torch", "torchvision", "cv2", "skimage", "fastapi")
foreach ($p in $pkgs) {
    & $py -c "import $p" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK  $p"
    } else {
        Write-Host "  GAGAL  $p belum terpasang atau tidak dapat dimuat"
        $ok = $false
    }
}

Write-Host "Memeriksa dataset..."
if (Test-Path "data\index.csv") {
    Write-Host "  OK  data\index.csv ditemukan"
} else {
    Write-Host "  BELUM  data\index.csv (dibuat setelah DIBaS diunduh)"
}

Write-Host ""
if ($ok) {
    Write-Host "Lingkungan siap." -ForegroundColor Green
} else {
    Write-Host "Ada yang perlu diperbaiki sebelum melanjutkan." -ForegroundColor Yellow
}
