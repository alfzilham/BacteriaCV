# ARCHITECTURE: Sistem Klasifikasi Bakteri Berbasis Computer Vision

## 1. Gambaran Umum
Pipeline linear lima tahap. Tahap pelatihan dan tahap inferensi berbagi modul pra-pemrosesan dan backbone yang sama, tetapi tahap pelatihan menambahkan lookup table dan class weighting.

```
[Citra mikroskop]
      |
      v
[1. Pra-pemrosesan] --- resize, normalisasi, segmentasi morfologis, watershed, augmentasi (latih saja)
      |
      v
[2. Backbone ResNet-50] --- dibekukan, pretrained ImageNet
      |
      v
[3. Vektor fitur 2048-d]
      |          \
      v           v
[4a. Head A]   [4b. Head B]
 softmax, 2     sigmoid, 2
 bentuk sel     status Gram
      |          |
      v          v
[5. Keluaran: label + confidence + visualisasi segmentasi]
```

Pada tahap pelatihan, label target diambil dari lookup table [L] sebelum masuk ke loss.

## 2. Komponen

### C1. Modul Pra-pemrosesan
- Input: citra RGB 2048 × 1532 piksel.
- Langkah: resize ke 224 × 224; normalisasi intensitas; segmentasi morfologis (erosi, dilasi, watershed); augmentasi hanya pada tahap latih (rotasi, flip, zoom, pergeseran).
- Output: tensor 3 × 224 × 224.
- Dependensi: OpenCV, NumPy, scikit-image.

### C2. Backbone
- ResNet-50 dari torchvision, bobot ImageNet.
- Seluruh parameter dibekukan pada versi awal.
- Output: vektor 2048 dimensi setelah global average pooling.

### C3. Classification Head
- Head A: Linear(2048, 2), aktivasi softmax, loss cross-entropy berbobot.
  Dua kelas: cocci dan bacilli. Bentuk spiral ada pada enum bentuk sebagai
  penanda kelas yang tidak terisi, karena DIBaS tidak memuat spesies spiral.
- Head B: Linear(2048, 2), aktivasi sigmoid, loss binary cross-entropy berbobot.
  **Hanya kolom keluaran kedua Head B yang masuk loss BCE.** Kolom keluaran
  pertama tidak pernah diregularisasi oleh loss; ia hanya terpengaruh weight
  decay, sehingga nilainya tidak membawa informasi yang dipelajari.
  Konsekuensinya, probabilitas kelas positif Head B adalah
  `sigmoid(logit kolom kedua)`, **bukan** softmax atas dua kolom. Memakai
  softmax akan menggabungkan logit yang dilatih dengan logit yang tidak
  dilatih, sehingga menambah noise dan menurunkan kalibrasi, bukan
  memperbaikinya. Hal ini terverifikasi: memakai softmax menurunkan akurasi
  validasi dari 1,0000 menjadi 0,9926 dan menambah citra berbanding rendah
  pada data uji dari 0 menjadi 1. Ambang klasifikasi memakai
  `sigmoid(logit) > 0,5`.
  Implementasinya ada di `train.py` baris 599 untuk loss BCE Head B dan
  `train.py` baris 363 untuk ambang klasifikasi.
- Bobot kelas dihitung dari frekuensi pada data latih.

### C4. Lookup Table [L]
- Memetakan 32 label spesies DIBaS ke pasangan (bentuk, status Gram).
  Candida albicans tidak dimasukkan karena dikeluarkan dari pelatihan.
- Digunakan hanya pada tahap pelatihan.
- Citra yang label spesiesnya tidak ada di tabel dibuang.
- Pemetaan nama spesies disimpan di `species_map.py`, bukan ditebak dari nama berkas.
  Enam nama arsip DIBaS mengandung salah ketik, dan dua arsip berbeda
  (`Lactobacillus.jehnsenii` dan `Lactobacillus.johnsonii`) menunjuk spesies taksonom
  yang sama, sehingga keduanya dibedakan lewat ID internal.

### C5. Modul Evaluasi
- Menghitung F1-score makro dan akurasi per head dari confusion matrix.
- Membagi data 70:20:10 secara acak per citra, dengan validasi silang lima lipatan.
- Lima lipatan hanya dibagikan di dalam 70% data latih. Data validasi (20%) dipakai untuk
  early stopping dan pemilihan checkpoint, data uji (10%) hanya dipakai sekali di akhir.
  Bila lipatan mencakup data validasi, maka data validasi ikut menjadi data latihan pada
  sebagian lipatan dan muncul optimistic bias pada metrik validasi.
- Lipatan diseimbangkan per spesies, bukan secara global. Spesies dengan 20 citra train
  menyumbang 3/3/3/3/2 ke lima lipatan, sedangkan spesies dengan 16 citra train
  menyumbang 4/3/3/3/3. Akibatnya ukuran absolut lipatan tidak sama, misalnya
  104/96/96/96/75. Yang dijaga adalah stratifikasi: setiap spesies tetap muncul
  di setiap lipatan, sehingga perbandingan antar lipatan tetap unbiased.

### C6. Modul Keluaran
- Menghasilkan label dan confidence score per head.
- Menghasilkan visualisasi segmentasi sebagai kontur area bakteri di atas citra asli, bukan kontur sel individual. Alasannya di SPEC bagian 8 butir 5: segmentasi memisahkan kelompok bakteri, bukan sel, dan elongasi objek tercampur susunan sel.

### C7. Antarmuka Web
- Unggah citra, jalankan inferensi, tampilkan hasil dan visualisasi.
- Backend: FastAPI. Frontend: halaman HTML tunggal.

## 3. Alur Data

### 3.1 Pelatihan
1. Muat citra dan label spesies dari DIBaS.
2. Terapkan lookup table [L] untuk memperoleh label bentuk dan Gram.
3. Buang citra yang tidak terpetakan.
4. Pra-pemrosesan dengan augmentasi.
5. Forward pass backbone sekali per citra, simpan vektor fitur ke cache.
6. Latih kedua head pada fitur beku. Hitung loss berbobot, perbarui parameter head saja.
7. Evaluasi pada data validasi, simpan checkpoint terbaik.
8. Evaluasi data uji sekali di akhir, setelah bobot terbaik dipulihkan.

Langkah 5 dan 6 dipisah karena backbone dibekukan. Forward pass pada 224 x 224 memakan sekitar 76 ms per citra; menjalankannya di dalam setiap epoch membuat 200 epoch memakanpuluhan menit tanpa mengubah vektor fitur. Cache kunci invalidated oleh nama berkas, ukuran, dan waktu modifikasi setiap citra, sehingga mengubah satu citra sumber sudah cukup untuk membangun ulang cache.

### 3.2 Inferensi
1. Terima citra dari antarmuka.
2. Pra-pemrosesan tanpa augmentasi.
3. Forward pass, lalu prediksi kedua head.
4. Hasil dikirim ke antarmuka.
Lookup table tidak digunakan pada tahap ini.

## 4. Struktur Direktori
```
BacteriaCV/
├── bacteriacv/
│   ├── paths.py                      # lokasi folder, tanpa path absolut
│   ├── preprocess.py                 # C1
│   ├── model.py                      # C2, C3
│   ├── label_map.py                  # C4
│   ├── train.py
│   ├── evaluate.py                   # C5
│   ├── infer.py
│   ├── visualize.py                  # C6
│   └── datasets/
│       ├── species_map.py            # 33 spesies DIBaS ke nama kanonik
│       ├── download.py               # unduh arsip + manifest SHA-256
│       ├── extract.py                # ekstraksi per spesies + verifikasi
│       └── build_index.py            # index.csv + pemeriksaan kebocoran
├── app/
│   ├── main.py                       # C7 backend
│   └── static/index.html             # C7 frontend
├── data/
│   ├── raw/
│   │   ├── zips/                     # arsip ZIP asli
│   │   ├── images/                   # citra TIFF per spesies
│   │   └── zips_manifest.csv         # SHA-256 tiap arsip
│   ├── unreadable.csv                # citra rusak, sudah rusak di arsip asal
│   ├── index.csv                     # path, spesies, split, fold
│   └── features/                     # cache fitur backbone, artefak turunan
├── checkpoints/                      # heads.pt, training_report.json, evaluation.json
├── docs/                             # SPEC, ARCHITECTURE, DESIGN, PRD, CONTEXT
├── scripts/                          # setup_env.ps1, check_env.ps1
├── tests/
└── AGENT.md
```

Struktur memakai paket `bacteriacv` dengan titik awal `bacteriacv.`. Alasan memakai paket, bukan folder `src` datar: modul dataset saling bergantung (`species_map` dipakai `download`, `extract`, dan `build_index`), dan import relatif antar paket lebih rapi daripada menumpuk modul di satu folder.

## 5. Keputusan Arsitektur
| ID | Keputusan | Alasan |
|----|-----------|--------|
| A1 | Satu backbone, dua head | Sesuai laporan Bab III dan IV.4; menghemat komputasi |
| A2 | Backbone dibekukan | Muat di GPU konsumen; mengurangi risiko overfitting |
| A3 | Klasifikasi per citra | Sesuai label DIBaS; tidak perlu segmentasi di inferensi |
| A4 | Watershed sebagai pra-pemrosesan | Mengurangi sel bertumpuk tanpa mengubah label |
| A5 | PyTorch | Modifikasi head dan loss ganda lebih eksplisit |
| A6 | Class weighting, bukan oversampling | Tidak menduplikasi citra; mencegah hasil validasi semu |
| A7 | Lipatan hanya di dalam data latih | Data validasi harus bebas dari data latihan agar early stopping tidak bias |
| A8 | Nama spesies disimpan di satu tabel | Enam nama arsip DIBaS salah ketik, dua arsip menunjuk spesies taksonom sama |
| A9 | SHA-256 tiap arsip dicatat | Server asal memakai sertifikat TLS kedaluwarsa, jadi integritas dibuktikan lewat hash |

## 6. Titik Ekstensi
- Fine-tuning penuh: buka pembekuan backbone pada `model.py`.
- Klasifikasi per sel: tambah tahap pemotongan objek setelah watershed, lalu jalankan inferensi per objek.
- Dataset lain: ganti `label_map.py` dan sediakan loader baru.

## 7. Penanganan Error
- Citra tidak terbaca: kembalikan pesan error, jangan hentikan layanan.
- Watershed gagal menemukan objek: lanjutkan dengan citra asli dan catat peringatan.
- Confidence rendah: tampilkan hasil dengan penanda ketidakpastian, bukan menyembunyikannya.
- Unduhan gagal: coba ulang tiga kali, lalu lempar error. Sisa unduhan sementara dibersihkan
  tanpa menutupi error asli.
- Citra di luar root proyek: tolak eksplisit, jangan tulis path absolut ke `index.csv`.
- Spesies dengan citra kurang dari 15 berkas: verifikasi ekstraksi gagal.

## 8. Catatan Integritas Data

Server asal DIBaS, `doctoral.matinf.uj.edu.pl`, memakai sertifikat TLS yang sudah kedaluwarsa.
Unduhan hanya dapat dilakukan tanpa verifikasi sertifikat. Untuk menutup celah tersebut, setiap
arsip di-hash SHA-256 dan hasilnya dicatat di `data/raw/zips_manifest.csv`. Hash tersebut yang
dipakai sebagai bukti provenance di laporan, bukan sertifikat server.

Jumlah berkas diverifikasi langsung terhadap arsip, hasilnya 692 berkas. Angka 660 pada paper
asli adalah perkiraan yang tidak sesuai isi arsip.

Candida albicans dikeluarkan dari pelatihan. Tiga berkas rusak juga dibuang, sehingga data
efektif 669 citra terbaca dari 32 spesies. Alasan dikeluarkan dua hal: jamur bukan bakteri,
dan ukuran sel jamur membuat kelas tersebut mudah dipisahkan sehingga F1-score tidak
mewakili kemampuan yang diukur.

Tiga berkas yang dibuang sudah rusak di dalam arsip DIBaS. SHA-256 berkas di disk identik
dengan entri arsip, jadi kerusakan berasal dari sumber, bukan dari proses ekstraksi. Daftar
lengkap beserta alasannya tercatat di `data/raw/unreadable.csv`.
