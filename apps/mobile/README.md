# BacteriaCVMobile

Aplikasi Android untuk klasifikasi bentuk sel dan status Gram bakteri,
**sepenuhnya on-device**. Tidak ada backend.

Status repo: **fase spike teknis**. Yang ada di sini bukan aplikasi,
melainkan pengukuran untuk menjawab empat pertanyaan sebelum satu baris
kode aplikasi ditulis. Jawaban dan semua angkanya ada di
[`docs/SPIKE-REPORT.md`](docs/SPIKE-REPORT.md).

---

## Yang sudah diketahui

- Model: ResNet50 ImageNet V2 yang dibekukan, dua head `Linear(2048, 2)`
- Kuantisasi: bobot float16, komputasi float32
- Target hanya Google Play, package id `alfzilham.ilham.bacteriacv_mobile`
- Preprocessing `cv2.INTER_AREA` tidak punya padanan di ONNX maupun
  TFLite, jadi diimplementasikan di Dart dan diuji terhadap cv2

## Yang belum diketahui

Lihat `docs/SPIKE-REPORT.md`. Empat pertanyaan yang diuji:

| | Pertanyaan |
|---|---|
| P1 | Apakah TIFF 2048 x 1532 bisa dibaca di Android, dan berapa lama |
| P2 | Apakah model bisa dikonversi ke ONNX dengan angka yang sama |
| P3 | Berapa lama inferensi di perangkat nyata, dan seberapa besar modelnya |
| P4 | Apakah segmentasi OpenCV bisa jalan on-device |

---

## Struktur

```
.githooks/    gatewa commit (audit karakter teks, dilacak di repo)
tools/        audit karakter + self-test + kontrol sabotase
docs/         sumber kebenaran, termasuk SPIKE-REPORT.md
lib/          kode Dart
  main.dart     entry point sementara, hanya pembuka layar spike
  spike/        layar spike, DIHAPUS setelah fase ini selesai
spike/        alat spike di luar paket aplikasi
  scripts/      konversi model dan perbandingan numerik
  artifacts/    hasil generate, di-ignore, boleh dihapus
android/      hasil flutter create --platforms android
```

## Batas keras

1. Tidak ada backend. Inferensi on-device.
2. Tidak ada file model yang masuk git. Bobot tinggal di project
   BacteriaCV dan dibaca read-only.
3. Arsitektur model dikunci: ResNet50, dua head Linear(2048, 2), fp16.
4. Tidak ada path absolut di kode atau dokumen.
5. Package id dikunci, tidak bisa diubah setelah rilis pertama.
6. Target hanya Android.

Detail lengkap ada di [`AGENTS.md`](AGENTS.md).

## Gatewa commit

```sh
git config core.hooksPath .githooks
```

Setiap commit menjalankan audit karakter non-Latin untuk seluruh repo
plus self-test yang membuktikan audit itu masih bisa gagal.

Perintah yang dijalankan spike ini:

```sh
python tools/check_text_encoding.py
python tools/tests/test_text_encoding_audit.py
```

## Menjalankan spike

```sh
flutter pub get
flutter run                     # pilih perangkat R9RYC0839YH
flutter test                    # uji INTER_AREA Dart terhadap acuan cv2
```

Model ONNX tidak ada di repo. Untuk menjalankan bagian P2 dan P3,
model harus dibangun lebih dulu:

```sh
python -m venv spike/.venv-tflite
spike/.venv-tflite/Scripts/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
spike/.venv-tflite/Scripts/pip install onnx onnxruntime onnxscript onnxconverter_common opencv-python-headless

spike/.venv-tflite/Scripts/python spike/scripts/build_onnx_model.py
spike/.venv-tflite/Scripts/python spike/scripts/compare_pytorch_fp16.py
```

`compare_pytorch_fp16.py` juga menulis berkas acuan yang dipakai
`flutter test` untuk membandingkan `INTER_AREA` Dart terhadap cv2.

Salinan model perlu ditaruh ke `assets/spike/model.onnx` sebelum
`flutter run` bisa memuatnya. Folder itu di-ignore, dan isinya tidak
pernah masuk git.