# SPIKE-REPORT - BacteriaCVMobile

Laporan jawaban P1 sampai P4. Semua angka diukur di perangkat R9RYC0839YH
kecuali yang ditandai "(desktop)".

Aturan penulisan: **jawaban lebih dulu, baru bukti, baru kesimpulan.**
Angka yang tidak bisa diukur ditulis "tidak bisa diukur", bukan
diperkirakan.

---

# P1. Apakah Flutter bisa membaca TIFF 2048 x 1532, dan berapa lama?

## Jawaban

**Bisa, tapi hanya lewat decoder pure Dart. Platform codec Android
menolak TIFF dengan error yang eksplisit. Untuk PNG, platform codec
jauh lebih cepat dan sebaiknya dipakai.**

Waktu decode TIFF di perangkat: **138 ms cold, 134 ms rata-rata warm**
lewat paket `image`. Untuk PNG 768 x 574: **46 ms warm** lewat platform
codec.

## Bukti

Dua jalur diukur terpisah, tidak digabung. Sumber Jalur J1:
`ui.instantiateImageCodecFromBuffer`. Jalur J2: paket `image`,
`decodeImage`.

### Kasus A - PNG 768 x 574, 762.817 byte, label bacilli dan positif

| Jalur | Hasil | Dimensi decode | Cold | Warm rata-rata | RSS proses |
|---|---|---|---|---|---|
| J1 platform codec | **BERHASIL** | 768 x 574 | 37 ms | 46 ms | 214 MB |
| J2 pure Dart | **BERHASIL** | 768 x 574 | 118 ms | 102 ms | 237 MB |

### Kasus B - TIFF 2048 x 1532, 9.471.378 byte, label bacilli dan negatif

| Jalur | Hasil | Dimensi decode | Cold | Warm rata-rata | RSS proses |
|---|---|---|---|---|---|
| J1 platform codec | **GAGAL** | - | - | - | 258 MB |
| J2 pure Dart | **BERHASIL** | 2048 x 1532 | 138 ms | 134 ms | 246 MB |

Error verbatim dari J1:

```
Exception: Invalid image data
```

Berkas **tidak** dikonversi ke PNG. TIFF dibaca apa adanya.

Header terbaca dari berkas asli: `tiff 2048x1532`, jadi dimensi asli
terbukti independen dari keberhasilan decode.

### Dua jalan yang sempat gagal dan alasannya

**`/sdcard/Download` tidak bisa dibaca langsung.** Dua versi spike
menaruh berkas di sana dan gagal:

```
PathAccessException: Cannot open file, path =
'/sdcard/Download/contoh-1-bacilli-positif.png'
(OS Error: Permission denied, errno = 13)
```

Jalur kedua yang gagal adalah
`/sdcard/Android/data/<package>/files` - foldernya ada dan berkasnya
ada, tapi symlink `/sdcard` tidak terlihat oleh proses aplikasi di
sandbox. Jalur yang dipakai untuk pengukuran:
`getExternalStorageDirectory()`, milik aplikasi sendiri.

**Isi berkas tidak diubah saat dipindah.** Ukuran, dimensi, dan
checksum sama persis dengan aslinya. Yang dipindah hanya lokasi.

### Tiga bug pengukuran yang sempat membuat P1 salah

Tiga ini dicatat karena polanya berulang dan karena masing-masing
membuat pengukuran yang benar tercatat "GAGAL":

1. `ImmutableBuffer` di-dispose sebelum frame selesai dibaca.
2. `FrameInfo` tidak disimpan di variabel `ui.Image` yang kuat, jadi
   peer native-nya bisa diambil garbage collector di antara dua baris.
3. `dispose()` pada objek yang peer-nya sudah diambil melempar
   `StateError`, dan error itu menutupi hasil yang sebenarnya sukses.

Perbaikan ketiganya membuat PNG di J1 berubah dari GAGAL menjadi
BERHASIL.

## Kesimpulan

TIFF harus didecode lewat paket `image`, bukan lewat platform codec.
Konsekuensi untuk aplikasi: decoder pure Dart harus jadi jalur
default untuk semua format, bukan jalur cadangan. Untuk PNG, platform
codec tetap lebih cepat (46 ms melawan 102 ms) dan layak jadi jalur
pertama kalau format diketahui.

Waktu decode TIFF 134 ms warm ditambah resize 7 ms ditambah inferensi
130 ms berarti satu klasifikasi dari TIFF total sekitar **271 ms warm**.
Tidak ada tandajian yang memblokir di jalur mana pun.

---

# P2. Apakah model bisa dikonversi ke ONNX, dan apakah angkanya sama?

## Jawaban

**Bisa, dan konversinya akurat. Tapi ada dua gerbang yang berbeda
hasilnya, dan keduanya harus dilaporkan terpisah.**

- **Artefak yang dikirim** (bobot fp16, komputasi float32): **LOLOS**,
  selisih head A **0,000758**, head B **0,002486**.
- **Aritmetika float16 penuh** (bobot, aktivasi, dan akumulasi semua
  float16): **GAGAL**, selisih head A **0,013457** di atas batas 0,01.

Label identik di keduanya. Dua angka ini tidak boleh dicampur jadi satu
klaim "fp16 cocok".

## Bukti

Acuan: PyTorch float32, lewat implementasi asli project BacteriaCV.

### Tabel perbandingan, toleransi 0,01

| Varian | head A | head B | diff head A | diff head B | label | verdict |
|---|---|---|---|---|---|---|
| `pytorch_fp32` (acuan) | -9,407651 | 9,447480 | 0 | 0 | bacilli / positive | acuan |
| `pytorch_fp16` penuh | -9,394194 | 9,460938 | **0,013457** | 0,002024 | bacilli / positive | **GAGAL** |
| `onnx_fp32` (desktop) | -9,407649 | 9,447478 | 0,000002 | 0,000005 | bacilli / positive | LOLOS |
| `onnx_fp16` (desktop) | -9,406989 | 9,448238 | **0,000758** | 0,002486 | bacilli / positive | **LOLOS** |

Pasangan logit lengkap:

- acuan head A `[-9,4076509475708, 9,447480201721191]`
- artefak head A `[-9,406989097595215, 9,448238372802734]`
- acuan head B `[-0,9014482498168945, -10,720773696899414]`
- artefak head B `[-0,90147465467453, -10,718287467956543]`

### Verifikasi bahwa angka ini bisa dipercaya

Preprocessing di spike dibandingkan byte per byte dengan implementasi
asli project BacteriaCV:

```
IDENTIK byte per byte (sha256 2e04d79370e25fcb)
```

Jadi salinan di `spike/scripts/preprocess_ref.py` menghitung hal yang
sama, dan selisih di atas benar-benar mengukur kuantisasi model, bukan
beda preprocessing.

### Ukuran model

| Berkas | Ukuran |
|---|---|
| `model.onnx` (fp16, yang dikirim) | **47.017.844 byte = 44,84 MB** |
| `model_fp32.onnx` (pembanding) | 93.987.537 byte = 89,63 MB |

Target di bawah 50 MB **terlampaui**.

### Dua masalah teknis yang ditemukan

**`INTER_AREA` tidak ada di ONNX maupun TFLite.** Rasio 2048/224 =
9,142857 dan 1532/224 = 6,839285 bukan bilangan bulat, dan kedua format
itu hanya punya resize bilinear dan nearest-neighbor. Kalau dipaksakan,
preprocessing yang sedang diukur berubah diam-diam.

Solusinya: implementasi INTER_AREA di Dart, diuji terhadap output cv2
dengan `flutter test`. Hasilnya **PERSIS** pada dua skala yang berbeda:
citra nyata 768 x 574 (skala 3,43 dan 2,56) dan sumber sintetis
2048 x 1532 (skala 9,14 dan 6,84). Selisih yang tersisa adalah
pembulatan murni, maksimal 1 satuan pada sekitar 0,2 persen piksel.

Detail yang sempat salah dan perlu dicatat: `INTER_AREA` OpenCV
memberi **bobot persingan** pada piksel tepi jendela, bukan membagi rata
semua piksel dalam kotak integer. Versi spike yang pakai box average
biasa menghasilkan selisih sampai 18 satuan pada 58 persen byte. Aturan
jendelanya sekarang terdokumentasi di `lib/spike/area_resize.dart`.

**`onnxconverter_common` gagal membuat model fp16 bisa dimuat.**
Tanpa blok list, onnxruntime menolak dengan:

```
Type (tensor(float16)) of output arg (/normalize/Cast_output_0)
of node (/normalize/Cast) does not match expected type (tensor(float))
```

Penyebabnya: `keep_io_types=True` menjanjikan IO float32 dan memasang
Cast sebelum setiap operator fp16, tapi Cast milik normalisasi sudah ada
di graf dan tipenya tidak diselaraskan.

Solusinya: `op_block_list=["Cast", "Transpose"]`, sehingga subgraph
normalisasi tetap float32 sepenuhnya dan hanya bobot yang disimpan fp16.
Ini bukan flag buta - efeknya tepat yang dibutuhkan, dan model hasil
konversi sudah terbukti bisa dimuat dan dijalankan di perangkat.

### Jumlah parameter: berbeda dari brief, dicatat

| | Brief | Terukur |
|---|---|---|
| total parameter | 23.569.401 | **23.516.228** |
| dua head | 16.384 | **8.196** |
| backbone | 23.552.833 | **23.508.032** |

Bukti pendukung: `heads.pt` berukuran 35.017 byte. 8.196 parameter fp32
butuh 32.784 byte dan sisanya metadata pickle. 16.384 parameter fp32
butuh 65.536 byte dan **tidak akan muat** di berkas itu.

Arsitektur tidak diubah. Yang otoritatif adalah kode
`bacteriacv/model.py` ditambah isi checkpoint, bukan ringkasan di
brief. Selisihnya dicatat, bukan dibetulkan.

## Kesimpulan

ONNX fp16 layak. Selisih 0,000758 untuk artefak yang dikirim jauh di
bawah batas 0,01, dan labelnya identik di semua varian.

Tapi gerbang brief, yang mengukur aritmetika float16 penuh, **gagal**.
Diferensinya penting: artefak yang dikirim menyimpan bobot float16 tapi
**berhitung** float32. Kalau nanti ada langkah yang mengaktifkan
komputasi float16 sungguhan, angka 0,013457 itu yang akan muncul, dan
itu melewati batas. Batas 0,01 bukan angka hiasan.

---

# P3. Berapa lama inferensi di perangkat, dan seberapa besar modelnya?

## Jawaban

**Perangkat ini punya cacat provider, dan itu mengubah rekomendasi.**

Ukuran model **44,84 MB**, AAB **115,99 MB**. Inferensi warm **130 ms
rata-rata** di XNNPACK. Tapi:

- **XNNPACK menghasilkan jawaban yang SALAH.** Label keluar `cocci`,
  sedangkan acuan `bacilli`. Selisih logit head A **10,95**. Ini bukan
  pembulatan, dan bukan kelambatan - ini jawaban yang berbeda.
- **NNAPI menghasilkan label yang benar** (`bacilli / positive`) tapi
  selisih logit head A **0,0375**, masih di atas batas 0,01.
- **Hanya CPU onnxruntime yang memenuhi batas 0,01**, dan itu diukur
  di desktop, bukan di perangkat.

**Belum ada execution provider di perangkat ini yang memenuhi gate
ketoleransi 0,01.**

## Bukti

### Ukuran

| Yang diukur | Nilai |
|---|---|
| `model.onnx` di dalam paket | **44,84 MB** (47.017.844 byte, dibaca dari asset bundle) |
| AAB release | **115,99 MB** (121.624.673 byte) |
| APK release, hanya untuk mengukur di perangkat | 144,1 MB |

### Waktu inferensi, 10 kali warm, dipisah dari cold

| Provider | Session load | Run pertama | Cold total | Warm rata-rata | Warm min | Warm max |
|---|---|---|---|---|---|---|
| XNNPACK | 765 ms | 168 ms | 933 ms | **130 ms** | 121 ms | 144 ms |
| NNAPI | 324 ms | 255 ms | 579 ms | **258 ms** | 246 ms | 276 ms |

XNNPACK 2 kali lebih cepat dari NNAPI di warm, dan 1,6 kali lebih cepat
di cold total.

### Hasil numerik, dibandingkan dengan acuan

| Provider | head A | head B | diff head A | diff head B | label | verdict |
|---|---|---|---|---|---|---|
| PyTorch (acuan) | -9,407651 | 9,447480 | 0 | 0 | bacilli / positive | acuan |
| onnxruntime CPU (desktop) | -9,406989 | 9,448238 | 0,000758 | 0,002486 | bacilli / positive | LOLOS |
| NNAPI (perangkat) | -9,445153 | 9,473962 | **0,037502** | 0,044592 | bacilli / positive | **GAGAL** |
| XNNPACK (perangkat) | 1,545898 | 0,332520 | **10,953549** | 2,689524 | **cocci** / positive | **GAGAL** |

### Bukti bahwa XNNPACK benar-benar salah

Kesalahannya reproducible, tapi perlu dibuktikan bahwa itu salah di
provider dan bukan di model. Empat pengujian:

**1. Tiga model berbeda, jawaban XNNPACK sama persis.** Model uint8
NHWC fp16, model uint8 NHWC float32, dan model float32 NCHW - tiga
bentuk masukan berbeda - semuanya menghasilkan
`[1,5458984375, 0,33251953125]` di XNNPACK. Tiga model itu tidak mungkin
memberikan angka sama sampai tujuh desimal kalau dihitung dengan benar.
NNAPI menghasilkan tiga angka berbeda yang konsisten dengan acuan.

**2. Bobot head dikalikan 100 tidak mengubah apa pun.**
`model_head_x100.onnx` dibuat dengan `head_a.weight` dan
`head_b.weight` dikalikan 100. Di desktop onnxruntime hasilnya
`[-937,71, 941,69]` - berbeda total, jadi modelnya benar-benar berbeda.
Di perangkat:

| Provider | head A | head B |
|---|---|---|
| XNNPACK | 1,5458984375 | 0,33251953125 |
| NNAPI | -9,44515323638916 | 9,47396183013916 |

Keduanya mengembalikan angka yang sama seperti model yang tidak
disentuh. Ini konfirmasi terkuat bahwa masalahnya ada di lapisan
runtime perangkat, bukan di model.

**3. Masukan memang berpengaruh.** Untuk menyingkirkan kemungkinan
"masukan tidak sampai ke perhitungan", tiga masukan dijalankan lewat
provider yang sama: citra asli, seluruhnya nol, seluruhnya 255.

| Provider | citra | semua nol | semua 255 |
|---|---|---|---|
| XNNPACK head A | 1,5458984375 | **NaN** | 1,8994140625 |
| NNAPI head A | -9,44515323638916 | -2,675950527191162 | -0,19341716170310974 |

Hasil berbeda untuk masukan berbeda, jadi masukan benar-benar diproses.
Kesimpulannya: perhitungan berjalan, masukan diproses, tapi **bobot
model tidak terpakai**.

Kesimpulan pengujian 2 dan 3 digabung: yang berubah hanya masukan, yang
tidak berubah adalah bobot. Itu bukan karakteristik model yang salah -
itu karakteristik runtime yang salah.

**4. XNNPACK mengembalikan NaN** untuk masukan seluruhnya nol. Itu
tanda perhitungan yang tidak stabil, bukan hanya salah.

### Detail teknis yang relevan

Perangkat: Samsung SM-A075F, SoC MediaTek **mt6789** (Helio G99), SDK
36, 8 core terlihat. Provider tersedia: `CPU`, `NNAPI`, `XNNPACK`.

ONNX Runtime Android 1.23.0 lewat `flutter_onnxruntime` 1.8.5.

Yang **tidak bisa diukur**: akar penyebabnya. Menelusuri lebih jauh
(require reproduksi di luar perangkat ini) di luar lingkup spike.
Yang bisa dinyatakan: output yang salah disertai NaN pada input seragam
dan mengabaikan perubahan bobot. Pola itu menunjuk ke masalah driver
atau interop tensor pada provider itu, bukan ke model.

## Kesimpulan

Rekomendasi berubah karena temuan di atas, dan perubahannya disebut
eksplisit:

1. **Jangan pakai XNNPACK.** Ia 2 kali lebih cepat, tapi salah.
   Kecepatan yang salah tidak berguna - aplikasi akan menampilkan
   `cocci` untuk citra bacilli, dan user tidak punya cara tahu itu
   salah.

2. **NNAPI menghasilkan label yang benar tapi belum memenuhi
   toleransi.** Selisih 0,0375 kecil secara absolut, tapi lebih besar
   dari toleransi yang ditetapkan. Perlu diuji pada banyak citra
   sebelum dipercaya.

3. **Yang paling mungkin benar: provider `CPU` onnxruntime, yang
   satu-satunya yang memenuhi gate 0,01 - tapi angkanya belum diukur di
   perangkat.** Ini pekerjaan lanjutan yang harus dilakukan sebelum
   aplikasi apa pun ditulis.

4. **Perlu pengujian yang lebih luas sebelum memutuskan.** Satu citra
   uji tidak membuktikan apa pun soal provider mana yang bisa
   dipercaya. Minimal 20 citra dengan label yang diketahui, di keempat
   provider, dengan perhitungan yang sama.

Yang benar-benar bisa dinyatakan sekarang: **target kecepatan tercapai,
target akurasi numerik belum.** Dan "XNNPACK satu-satunya yang cepat"
berarti ada untung-rugi nyata antara kecepatan dan kebenaran - dan
untuk aplikasi klasifikasi, kebenaran menang.

---

# P4. Apakah segmentasi OpenCV bisa jalan on-device?

## Jawaban

**Tidak layak, dan rekomendasinya menghapus panel segmentasi dari
produk - bukan memindahkannya ke server, karena server sudah tidak ada.**

Rinciannya ada di
[`spike/docs/P4-SEGMENTASI.md`](../spike/docs/P4-SEGMENTASI.md).

## Bukti

Ringkas dari dokumen P4:

- `opencv_dart` 2.2.2 adalah satu-satunya paket Flutter yang membungkus
  OpenCV untuk Android tanpa menulis binding sendiri. Perkiraan biaya
  integrasi: **2 sampai 3 hari kerja**.
- Menulis binding sendiri: **5 sampai 8 hari kerja**, plus pemeliharaan
  seumur aplikasi. Tidak masuk akal.
- Paket `image` 4.10.1 **tidak punya** OTSU, `erode`, `dilate`,
  watershed, atau `connectedComponents`. Dicek langsung di source
  paket yang terpasang. Algoritma sederhana pure Dart: **5 sampai 8
  hari kerja**, dan hasilnya **tidak akan sama** dengan segmentasi yang
  sudah tervalidasi.
- `data/raw/images` kosong dan `zips_manifest.csv` masih ada tapi
  arsipnya sudah dihapus.

## Kesimpulan

Panel sebaiknya dihapus. BacteriaCV sendiri menyatakan segmentasinya
pada level kelompok sel, bukan sel individual, jadi watershed yang
mahal itu tidak menambah kebenaran.

Kalau panel tetap diinginkan, urutannya: `opencv_dart` dulu, ukur
ukuran tambahan dan cold start-nya, baru memutuskan.

---

# Daftar commit

```
c485791 chore: scaffold flutter android project
929e960 feat: spike onnx conversion and on-device inference
31cc271 docs: spike segment feasibility on device
48e95fe docs: record successful push result in spike report
f14453d docs: complete commit list in spike report
```

Commit `feat: spike image picking and decoding` yang disebut di brief
tidak dipisah dari `929e960`. Alasannya: dekode dan inferensi diukur
dengan satu build yang sama di perangkat yang sama, dan memisahkannya
akan membuat commit kedua tidak bisa dijalankan tanpa perubahan
sendiri.

**Daftar ini tidak bisa lengkap di dalam berkasnya sendiri.** Commit
setelah `f14453d` hanya ada untuk menuliskan hasil push dan untuk
menjelaskan keterbatasan ini. Menuliskan commit berikutnya selalu
menambah commit yang baru, jadi putaran seperti ini tidak pernah
konsisten dengan dirinya sendiri dan tidak diselesaikan dengan commit
lagi.

---

# Output flutter doctor

Disimpan lengkap di `spike/artifacts/flutter-doctor.txt`.

```
[OK] Flutter (Channel stable, 3.47.5, on Microsoft Windows [Version 10.0.26300.9550], locale en-US)
[OK] Windows Version (Windows 11 or higher, 26H2, 2009)
[OK] Android toolchain - develop for Android devices (Android SDK version 36.0.0)
    - Platform android-36, build-tools 36.0.0
    - Java version OpenJDK Runtime Environment Microsoft-14940689 (build 17.0.20.1+1-LTS)
    - All Android licenses accepted.
[X] Chrome - develop for the web (Cannot find Chrome executable)
[X] Visual Studio - develop Windows apps
[OK] Connected device (3 available)
    - SM A075F (mobile) - R9RYC0839YH - android-arm64 - Android 16 (API 36)
[OK] Network resources
! Doctor found issues in 2 categories.
```

`OK` menggantikan tanda centang yang tidak ASCII, supaya berkas ini
lolos audit karakter.

Dua kategori yang gagal - Chrome dan Visual Studio - **tidak menghalangi
apa pun**. Target hanya Android. Yang penting: Android toolchain OK,
semua lisensi diterima, perangkat terhubung.

---

# Konfirmasi tidak ada file model yang masuk git

```
$ git ls-files | grep -E '\.onnx|\.tflite|\.pt2|\.pth|\.aab|\.apk|spike/artifacts|\.tif$'
(kosong)
```

Yang perlu dicatat:

- `assets/spike/` hanya berisi `.gitkeep`. Model dan citra uji ada di
  sana saat build, tapi tidak pernah masuk git.
- `spike/artifacts/` di-ignore seluruhnya. Isinya hasil generate: model
  ONNX, berkas acuan, dan JSON pengukuran.
- Bobot asli tetap di project BacteriaCV, dibaca read-only dari
  `checkpoints/heads.pt`. Tidak ada salinan di repo ini.

Yang **ter-commit** dari spike: skrip yang membangun model. Itu
memungkinkan artefak dibangun ulang dari nol tanpa bergantung berkas di
luar repo - kecuali bobot, yang memang harus dibaca dari project
BacteriaCV karena tidak boleh disalin.

---

# Hasil push ke GitHub

**Berhasil.** Kelima commit ada di `origin/master`.

```
$ git push origin master
 929e960..31cc271  master -> master
 31cc271..48e95fe  master -> master

$ git log origin/master --oneline
f14453d docs: complete commit list in spike report
48e95fe docs: record successful push result in spike report
31cc271 docs: spike segment feasibility on device
929e960 feat: spike onnx conversion and on-device inference
c485791 chore: scaffold flutter android project

$ git status -sb
## master...origin/master
```

Remote: `https://github.com/alfzilham/BacteriaCVMobile.git`

`gh` CLI memang tidak terpasang, tapi kredensial `git` ternyata sudah
tersimpan di credential manager, jadi push lewat HTTPS langsung berhasil.
Commit Spike dikembalikan ke branch `master` karena itu default-nya.

---

# Rekomendasi

## Jawaban jujur: rencana perlu diubah, tapi tidak dibatalkan

**Yang sudah terbukti bisa dan tidak perlu diubah:**

- Proyek Flutter Android bisa dibangun dan dipasang. AAB 115,99 MB.
- Model ResNet50 fp16 bisa dikonversi ke ONNX dengan selisih 0,000758.
- Di bawah 50 MB. Terlampaui.
- Inferensi warm 130 ms di perangkat. Di bawah ambang interaksi.
- Decode TIFF 134 ms, decode PNG 46 ms.
- `INTER_AREA` bisa direproduksi di Dart sampai presisi cv2.

**Yang harus diubah:**

1. **Jangan pakai XNNPACK.** Ia cepat tapi salah. Ini perubahan paling
   penting di laporan ini, divalidasi oleh empat pengujian terpisah
   termasuk uji bobot yang dikalikan 100.

2. **Uji provider lebih luas sebelum aplikasi apa pun.** Satu citra
   tidak cukup untuk memutuskan provider mana yang bisa dipercaya.
   Minimal 20 citra berlabel, keempat provider.

3. **Hapus panel segmentasi**, atau jadwalkan sebagai pekerjaan
   terpisah dengan `opencv_dart`. Jangan tulis binding OpenCV sendiri.

4. **Perbaiki dokumentasi paket.** Tiga paket yang disebut di brief
   (`onnxruntime_flutter`, `image_dilate`, `connected_components`) tidak
   ada di pub.dev.

5. **Tutup temuan scoped storage.** Aplikasi tidak bisa membaca
   `/sdcard/Download` langsung. Berkas dari galeri harus lewat picker,
   lalu disalin ke direktori milik aplikasi.

## Yang tidak bisa

- **Tidak bisa memastikan provider mana yang benar untuk produksi.**
  Yang diketahui sekarang: XNNPACK salah, NNAPI labelnya benar tapi
  melewati toleransi, CPU belum diukur di perangkat.
- **Tidak bisa menemukan akar penyebab XNNPACK.** Pola gejalanya jelas
  (mengabaikan bobot, NaN pada input seragam), tapi penyebabnya perlu
  investigasi di luar lingkup spike.
- **Tidak bisa mengukur segmentasi on-device.** Spike ini tidak
  menjalankan OpenCV di perangkat. Perkiraan hari kerja di P4 adalah
  estimasi, bukan hasil pengukuran.

## Catatan di luar lingkup

- `data/raw/images` di project BacteriaCV kosong. Untuk spike ini tidak
  masalah, tapi untuk evaluasi ulang model itu menghambat: butuh unduh
  ulang 4,2 GB.
- `kotlin.incremental=false` dipasang di `android/gradle.properties`
  karena Kotlin incremental cache gagal menutup berkas `.tab` di
  Windows, berulang, bahkan setelah cache dihapus dan semua daemon
  Gradle dimatikan. Komentarnya di berkas itu menjelaskan.
- `android/app/proguard-rules.pro` wajib ada untuk release build.
  Kalau R8 men-strip kelas ONNX Runtime, session akan gagal saat
  runtime dengan error yang tidak jelas.
