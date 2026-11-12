# setup_env.ps1
# Menyiapkan lingkungan virtual dan dependensi untuk proyek (mode CPU).
# Jalankan dari folder root proyek: .\scripts\setup_env.ps1

$ErrorActionPreference = "Stop"

Write-Host "[1/5] Memeriksa Python..."
$pyVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python tidak ditemukan. Pasang Python 3.10 atau lebih baru dari python.org." -ForegroundColor Red
    exit 1
}
Write-Host "      $pyVersion"

Write-Host "[2/5] Membuat virtual environment (.venv)..."
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

Write-Host "[3/5] Mengaktifkan virtual environment..."
& ".\.venv\Scripts\Activate.ps1"

Write-Host "[4/5] Memasang PyTorch versi CPU..."
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Host "[5/5] Memasang dependensi lainnya..."
python -m pip install opencv-python scikit-image numpy pandas scikit-learn pyyaml fastapi uvicorn python-multipart pytest

Write-Host ""
Write-Host "Verifikasi:" -ForegroundColor Green
python -c "import torch, cv2, skimage, fastapi; print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); print('opencv', cv2.__version__); print('skimage', skimage.__version__); print('fastapi', fastapi.__version__)"

Write-Host ""
Write-Host "Selesai. Aktifkan lingkungan dengan: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
