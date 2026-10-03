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
 softmax, 3     sigmoid, 2
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
- Head A: Linear(2048, 3), aktivasi softmax, loss cross-entropy berbobot.
- Head B: Linear(2048, 2), aktivasi sigmoid, loss binary cross-entropy berbobot.
- Bobot kelas dihitung dari frekuensi pada data latih.

### C4. Lookup Table [L]
- Memetakan 33 label spesies DIBaS ke pasangan (bentuk, status Gram).
- Digunakan hanya pada tahap pelatihan.
- Citra yang label spesiesnya tidak ada di tabel dibuang.
- Pemetaan nama spesies disimpan di `species_map.py`, bukan ditebak dari nama berkas.
  Lima nama arsip DIBaS mengandung salah ketik, dan dua arsip berbeda
  (`Lactobacillus.jehnsenii` dan `Lactobacillus.johnsonii`) menunjuk spesies taksonom
  yang sama, sehingga keduanya dibedakan lewat ID internal.

### C5. Modul Evaluasi
- Menghitung F1-score makro dan akurasi per head dari confusion matrix.
- Membagi data 70:20:10 secara acak per citra, dengan validasi silang lima lipatan.
- Lima lipatan hanya dibagikan di dalam 70% data latih. Data validasi (20%) dipakai untuk
  early stopping dan pemilihan checkpoint, data uji (10%) hanya dipakai sekali di akhir.
  Bila lipatan mencakup data validasi, maka data validasi ikut menjadi data latihan pada
  sebagian lipatan dan muncul optimistic bias pada metrik validasi.

### C6. Modul Keluaran
- Menghasilkan label dan confidence score per head.
- Menghasilkan visualisasi segmentasi (kontur sel di atas citra asli).

### C7. Antarmuka Web
- Unggah citra, jalankan inferensi, tampilkan hasil dan visualisasi.
- Backend: FastAPI. Frontend: halaman HTML tunggal.

## 3. Alur Data

### 3.1 Pelatihan
1. Muat citra dan label spesies dari DIBaS.
2. Terapkan lookup table [L] untuk memperoleh label bentuk dan Gram.
3. Buang citra yang tidak terpetakan.
4. Pra-pemrosesan dengan augmentasi.
5. Forward pass backbone, lalu kedua head.
6. Hitung loss berbobot, perbarui parameter head saja.
7. Evaluasi pada data validasi, simpan checkpoint terbaik.

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
│   └── index.csv                     # path, spesies, split, fold
├── checkpoints/
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
| A8 | Nama spesies disimpan di satu tabel | Lima nama arsip DIBaS salah ketik, dua arsip menunjuk spesies taksonom sama |
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

Jumlah citra diverifikasi langsung terhadap arsip, hasilnya 692 citra. Angka 660 pada paper
asli adalah perkiraan yang tidak sesuai isi arsip.

Candida albicans dikeluarkan dari pelatihan sehingga data efektif 672 citra dari 32 spesies.
Alasannya dua hal: jamur bukan bakteri, dan ukuran sel jamur membuat kelas tersebut mudah
dipisahkan sehingga F1-score tidak mewakili kemampuan yang diukur.
