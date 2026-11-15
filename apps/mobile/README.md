# BacteriaCV Mobile

Aplikasi Android untuk klasifikasi bentuk sel (cocci atau bacilli) dan status
Gram (positif atau negatif) dari satu citra mikroskop. Inferensi berjalan
**sepenuhnya di perangkat**. Tidak ada backend dan tidak ada permintaan jaringan.

Status: versi `0.1.0+1`, belum dirilis. Package id `alfzilham.ilham.bacteriacv_mobile`
dikunci dan tidak boleh diubah. Tombol unduh di situs web tetap berlabel demo
sampai ada APK bertanda tangan.

Hasil aplikasi ini adalah alat bantu, bukan diagnosis.

## Yang ada di aplikasi

- Memilih berkas citra (TIFF, PNG, atau JPG) lewat pemilih berkas sistem. TIFF
  tidak muncul di galeri Android, jadi galeri tidak dipakai.
- Batas ukuran berkas 20 MB, ditegakkan sebelum berkas didecode.
- Layar hasil: bentuk sel, status Gram beserta level keyakinan (high, medium,
  low), dan waktu proses. Persentase keyakinan sengaja tidak ditampilkan.
- Ambang level 0,8 dan 0,6, sama dengan situs web.
- Tema terang dan gelap, serta disclaimer "bukan diagnosis".

## Struktur

```
lib/
  main.dart, app.dart   titik masuk dan konfigurasi aplikasi
  controllers/          pengendali tema
  models/               classifier.dart (inferensi), area_resize.dart (INTER_AREA di Dart)
  theme/                token warna
  views/                layar upload, layar hasil, disclaimer, tombol tema
test/                   uji Dart (ambang level, resize, layar hasil, aksesibilitas, tema)
android/                proyek Android (Gradle Kotlin DSL)
assets/spike/           tempat model ONNX, isinya di-ignore git
```

## Model ONNX: disediakan terpisah

Berkas model **tidak ada di repo**. `*.onnx` di-ignore git, dan skrip yang
menghasilkan model juga tidak ada di repo ini. Aplikasi mengharapkan berkas ini:

```
apps/mobile/assets/spike/model_int8.onnx
```

Kontrak model yang dipakai kode:

| | Nama | Bentuk | Tipe |
|---|---|---|---|
| Masukan | `image_u8` | `[1, 224, 224, 3]` (NHWC) | uint8 |
| Keluaran | `head_a_logits` | `[1, 2]` | float32 |
| Keluaran | `head_b_logits` | `[1, 2]` | float32 |

Normalisasi ImageNet ada di dalam graf model. Resize INTER_AREA tidak punya
padanan di ONNX, jadi dilakukan di Dart (`area_resize.dart`).

### Asal-usul dan validasi model int8

Pernyataan ini sengaja jujur tentang apa yang belum diketahui:

- Asal-usul `model_int8.onnx` **tidak tercatat di repo**. Skrip kuantisasinya
  (`spike/scripts/quantize_int8.py`) hanya disebut namanya di arsip audit, dan
  skripnya sendiri tidak ada di monorepo ini.
- Laporan [SPIKE-REPORT.md](../../docs/v1-mobile-docs/report/SPIKE-REPORT.md)
  hanya memvalidasi model **fp16** terhadap PyTorch. Laporan itu tidak
  memvalidasi int8.
- Pengukuran terpisah (15 November 2026) membandingkan keluaran
  `model_int8.onnx` dengan pipeline API float32 pada 696 citra (692 TIFF DIBaS
  dan 4 contoh situs), memakai ONNX Runtime di CPU dengan masukan uint8 yang
  sama:

  | Ukuran | Bentuk sel | Status Gram |
  |---|---|---|
  | Label sama dengan float32 | 696 dari 696 | 696 dari 696 |
  | Selisih keyakinan maksimum | 0,28 | 0,24 |
  | Selisih keyakinan rata-rata | 0,007 | 0,009 |
  | Level keyakinan sama dengan float32 | 680 dari 696 | 690 dari 696 |

- Batas pengukuran itu: bukan XNNPACK di perangkat, tidak dibandingkan dengan
  label sebenarnya, dan hanya berlaku untuk berkas model yang diukur. Dua
  berkas DIBaS tidak terbaca dan dilewati.
- Karena keyakinan int8 tidak identik dengan float32, level high, medium, dan
  low bisa berbeda dari situs web untuk citra yang keyakinannya dekat ambang,
  walaupun ambangnya sama.

## Menjalankan

Prasyarat: Flutter 3.47 (CI memakai `3.47.x`) dan Android SDK.

```sh
cd apps/mobile
flutter pub get
flutter analyze --fatal-infos
flutter test
flutter run
```

`flutter test` tidak membutuhkan model ONNX. `flutter run` membutuhkan model
di `assets/spike/model_int8.onnx` seperti dijelaskan di atas.

## Build rilis

Build rilis membutuhkan kunci rilis milik Anda sendiri. Keystore dan
`android/key.properties` **tidak boleh** masuk repo (sudah di-ignore).

## Batas yang dijaga

1. Tidak ada backend. Inferensi di perangkat.
2. Tidak ada berkas model di git.
3. Package id tidak berubah.
4. Ambang level dan jalur inferensi tidak diubah tanpa membahasnya di issue,
   karena situs web, API, dan aplikasi ini memakai model yang sama.
