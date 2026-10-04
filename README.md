# BacteriaCV

BacteriaCV adalah proyek computer vision untuk klasifikasi bentuk sel bakteri dan status Gram dari citra mikroskop. Repositori saat ini berfokus pada penyiapan dataset DIBaS: pengunduhan arsip, ekstraksi citra, pemetaan spesies, dan pembangunan `data/index.csv` yang bebas kebocoran data.

## Status proyek

- Ekstraksi penuh menghasilkan 692 citra dari 33 label spesies.
- `Candida albicans` dikecualikan dari pelatihan karena jamur, bukan bakteri, sehingga tersisa 32 spesies yang dapat dilatih.
- `data/index.csv` memuat 672 citra: 469 train, 138 validation, dan 65 test.
- Lima lipatan hanya dibuat dari data train.
- Nama spesies dipetakan melalui `bacteriacv/datasets/species_map.py`, bukan ditebak dari nama berkas.
- Modul pelatihan dan antarmuka web belum menjadi bagian dari tahap dataset ini.

## Struktur utama

```text
BacteriaCV/
├── bacteriacv/
│   ├── config.py
│   ├── label_map.py
│   ├── paths.py
│   └── datasets/
│       ├── species_map.py
│       ├── download.py
│       ├── extract.py
│       └── build_index.py
├── data/
│   ├── index.csv
│   └── raw/
│       └── zips_manifest.csv
├── docs/
├── scripts/
└── tests/
```

## Persiapan lingkungan

Di PowerShell dari root repositori:

```powershell
.\scripts\setup_env.ps1
.\scripts\check_env.ps1
```

Atau gunakan Python environment yang sudah tersedia dan pasang dependensi proyek sesuai kebutuhan.

## Pemasangan

Butuh Python 3.11.0. Dari root repositori:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Indeks CPU untuk torch sudah tertangani di dalam `requirements.txt` lewat `--extra-index-url`, jadi tidak ada perintah tambahan. `torch` dan `torchvision` memakai local version suffix `+cpu` yang tidak ada di PyPI, dan tanpa indeks tambahan itu `pip install` gagal dengan `No matching distribution found`.

Aplikasi juga butuh folder `checkpoints/` yang berisi `heads.pt` dan `evaluation.json`. Tanpa `heads.pt` aplikasi berhenti saat start dengan `FileNotFoundError`. Tanpa `evaluation.json`, `/api/report` mengembalikan 404 "Evaluasi belum dijalankan".

## Menjalankan aplikasi

```sh
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

`${PORT:-8000}` disengaja. Railway menyuntikkan variabel `PORT` secara dinamis lalu mengarahkan trafik ke nomor itu, dan bentuk `:-8000` memakai nilai cadangan 8000 kalau `PORT` tidak ada, sehingga perintah yang sama jalan di server maupun di mesin lokal. Karena itu jangan menulis `--port 8000` polos di mana pun; port yang dipatok membuat Railway tidak mencapai aplikasinya. Bentuk `${PORT:-8000}` adalah ekspansi shell POSIX. Di PowerShell, setel portnya lebih dulu: `$env:PORT = "8123"`.

Periksa kesehatan:

```
http://127.0.0.1:8000/api/health
```

Jawaban yang benar saat model sudah termuat:

```json
{"status":"siap","checkpoint":"heads.pt","max_upload_bytes":20971520}
```

Batas unggahan 20 MB. Aplikasi menerima berkas lebih besar dari 5 MB, jadi jangan menulis 5 MB di mana pun. Format yang diterima: `png`, `jpg`, `jpeg`, `tif`, `tiff`.

Berkas `heads.pt` di root repositori bukan model yang dipakai dan sudah dihapus. Bobotnya berbeda dari `checkpoints/heads.pt` dan tidak pernah dibaca aplikasi. Yang dimuat selalu `checkpoints/heads.pt`.

## Pipeline dataset

Unduh 33 arsip DIBaS dan catat manifest SHA-256:

```powershell
python -m bacteriacv.datasets.download
```

Ekstrak citra dan verifikasi hasilnya:

```powershell
python -m bacteriacv.datasets.extract
python -m bacteriacv.datasets.extract --verify-only
```

Bangun index dataset:

```powershell
python -m bacteriacv.datasets.build_index
```

`data/index.csv` berisi path relatif, nama spesies, `species_id`, split, dan fold. Data validasi serta data uji memakai `fold=-1`.

Catatan: server asal DIBaS memiliki sertifikat TLS kedaluwarsa. Modul unduhan menggunakan koneksi tanpa verifikasi sertifikat untuk sumber tersebut dan mencatat SHA-256 setiap arsip di `data/raw/zips_manifest.csv` sebagai bukti provenance.

## Pengujian

```powershell
python -m pytest -q
```

Tes mencakup pemetaan spesies, lokasi data, unduhan, ekstraksi, pembagian dataset, pemeriksaan kebocoran, fold, dan path relatif.

## Dokumentasi

- [SPEC](docs/SPEC.md) — sumber kebenaran kebutuhan proyek.
- [ARCHITECTURE](docs/ARCHITECTURE.md) — arsitektur dan keputusan teknis.
- [DESIGN](docs/DESIGN.md) — desain antarmuka dan kontrak modul.
- [AGENT](AGENT.md) — aturan kontribusi dan audit.

## Lisensi

Proyek ini dirilis di bawah [MIT License](LICENSE).
