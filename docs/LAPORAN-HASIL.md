# Laporan Hasil Ekperimen

Semua angka di dokumen ini hasil eksperimen nyata pada dataset DIBaS yang
sudah diekstrak. Tidak ada angka proyeksi. Angka proyeksi 95% sampai 99% di
SPEC bagian 6 bukan syarat keberhasilan dan tidak dipakai di sini.

Tanggal eksekusi: 4 Oktober 2026.

## 1. Konfigurasi yang Dipakai

| Hyperparameter | Nilai | Sumber |
|----------------|-------|--------|
| Backbone | ResNet-50, `IMAGENET1K_V2`, dibekukan | `config.BACKBONE_NAME`, `BACKBONE_WEIGHTS` |
| Feature dim | 2048 | `config.FEATURE_DIM` |
| Head A | `Linear(2048, 2)` softmax, kelas `cocci` dan `bacilli` | `config.N_SHAPE_CLASSES` |
| Head B | `Linear(2048, 2)` sigmoid, kelas `positif` dan `negatif` | `config.N_GRAM_CLASSES` |
| Augmentasi | 4 varian per citra latih (1 asli + 3 rotasi/skala/geser/flip) | `config.AUGMENT_VARIANTS` |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 | `config.LEARNING_RATE`, `WEIGHT_DECAY` |
| Batch size | 32 | `config.BATCH_SIZE` |
| Batas epoch | 200 | `config.MAX_EPOCHS` |
| Early stopping | patience 20, min delta 1e-4 | `config.EARLY_STOPPING_PATIENCE`, `MIN_DELTA` |
| Seed pelatihan | 1337 | `config.TRAIN_SEED` |
| Seed index | 20260203 | `config.INDEX_SEED` |
| Resolusi segmentasi | 2048 x 1532 (asli) | `config.SEGMENT_SCALE` |
| Batas unggah | 20 MB | `config.MAX_UPLOAD_BYTES` |

## 2. Deviasi dari Rencana dan Alasannya

| Yang diubah | Rencana | Kenyataan | Alasan |
|----------------|--------|-----------|--------|
| Ekstraksi fitur | `preprocess` penuh | `preprocess_tensor` tanpa segmentasi | Segmentasi pada resolusi asli memakan 2,5 detik per citra. 2070 baris fitur akan memakan 86 menit tanpa memperbaiki metrik model, karena ARCHITECTURE bagian 3 menyatakan segmentasi tidak dibutuhkan inferensi. |
| Panel visualisasi | Lima panel | Satu panel dengan tombol alih lima tahap | DESIGN bagian 3 peringkat 3 menang atas CONTEXT peringkat 4. Yang dikirim ke browser lima panel, yang ditampilkan satu per satu. |
| Penyaringan luas objek | `regionprops` per region | `np.unique` plus `np.bincount` | Versi loop membandingkan seluruh citra berlabel untuk tiap region, 3,76 detik per citra. Mask hasil identik, hanya lebih cepat. |
| Label spesies | `strain` | spesies | Tidak ada strain pada DIBaS; yang ada spesies. |
| Jumlah epoch | 200 | 35, berhenti early stopping | Patience 20, epoch terbaik 15. Menjalankan 200 epoch tidak memperbaiki validasi. |

## 3. Data

- 692 berkas TIFF diarsipkan, 689 terbaca, 669 dipakai setelah
  `Candida albicans` dikeluarkan.
- Pembagian: 467 latih, 136 validasi, 66 uji.
- Setelah augmentasi: 1868 baris latih, 136 baris validasi, 66 baris uji.
- Distribusi latih setelah augmentasi: `cocci` 576, `bacilli` 1494;
  `positif` 1471, `negatif` 599.
- Tiga citra dibuang: `Listeria 0023` 0 byte, `Micrococcus 0021` dan `0023`
  TIFF rusak. Ketiganya terbukti sudah rusak di arsip asal lewat SHA-256.

## 4. Hasil Inference

| Metrik | Head A (bentuk) | Head B (Gram) |
|--------|-----------------|----------------|
| F1 makro validasi, epoch terbaik | 0,9817 | 1,0000 |
| F1 makro uji | 1,0000 | 1,0000 |
| Akurasi uji | 1,0000 | 1,0000 |
| Presisi uji per kelas | 1,0000 kedua kelas | 1,0000 kedua kelas |
| Recall uji per kelas | 1,0000 kedua kelas | 1,0000 kedua kelas |
| Support uji | `cocci` 19, `bacilli` 47 | `positif` 47, `negatif` 19 |
| Confidence rata-rata uji | 0,9812 | 0,9671 |

Checkpoint: `checkpoints/heads.pt`. Epoch yang dijalankan 35, epoch terbaik 15,
epoch 1 F1 bentuk 0,9554 dan F1 Gram 0,8921.

Data uji berisi 66 citra dari 32 spesies, sekitar dua citra per spesies. F1
makro 1,0000 berarti tidak ada satu pun kesalahan pada 66 citra. Itu bukan
berarti model sempurna pada data lain. Margin antar citra memang sempit:
confidence bentuk 0,9812 dan confidence Gram 0,9671, keduanya di atas 0,8.
Kesalahan pada citra di luar data uji tetap sangat mungkin terjadi.

Confidence Gram dihitung dari `sigmoid(logit)` pada kolom keluaran kedua.
Head B dilatih sebagai klasifier biner satu logit dengan `BCEWithLogitsLoss`,
sehingga `sigmoid(logit)` memang probabilitas kelas positif dan ambang 0,5
berlaku langsung. Kolom keluaran pertama tidak pernah masuk loss. Bukti:
mengubah kolom itu menjadi 99.999 tidak mengubah nilai loss sama sekali
sehingga kolom itu hanya mengalami weight decay dan tidak membawa informasi.
Karena itu softmax atas dua kolom keluaran tidak boleh dipakai untuk Head B.
Pengukuran pada 136 citra validasi justru menunjukkan softmax menurunkan
akurasi dari 1,0000 menjadi 0,9926 dan menambah citra berbanding rendah dari
satu menjadi dua.

## 5. Keterbatasan Segmentasi

Segmentasi morfologis berhasil memisahkan kelompok bakteri tetapi bukan sel
individual, karena sel bersentuhan pada citra Gram 100x. Elongasi median objek
untuk spesies kokus adalah 1,30 sampai 1,43 dan untuk spesies batang 1,64 sampai
1,76 di bawah metrik major/minor axis, namun rentang intraspesies 1,03 sampai
4,19 melampaui selisih antargrup 0,068, sehingga klasifikasi bentuk tidak dapat
divalidasi dari objek hasil segmentasi. Metrik berbasis luas (extent) tidak
menunjukkan pemisahan sama sekali.

Bukti ini terkonstrain dua arah. Elongasi objek mencampur bentuk sel dengan
susunan sel berupa klaster, rantai, dan tetrad, yang tidak dapat dipisahkan
oleh metode ini. Konsekuensinya panel visualisasi menampilkan kontur area
bakteri, bukan sel individual, dan selalu menuliskan keterbatasan ini di dalam
citra panel.

Konstanta segmentasi tidak diubah setelah eksperimen ini. `SEGMENT_MIN_PEAK_DISTANCE`
tetap 15. Mengubahnya untuk memperlebar celah antargrup akan mengejar noise,
karena rentang intraspesies sudah lebih lebar dari selisih antargrup. Kalau
constants ditinjau ulang, kriterianya kestabilan: pilih nilai yang menghasilkan
rentang intraspesies terkecil supaya kontur pada panel tidak berubah-ubah antar
citra.

## 6. Kecepatan

| Tahap | Waktu |
|-------|-------|
| Baca TIFF 2048 x 1532 | 69 ms |
| Resize dan normalisasi | 25 ms |
| Segmentasi resolusi asli | 2.502 ms |
| Forward pass ResNet-50 224 x 224 | 76 ms |
| Ekstraksi 2070 baris fitur, sekali | sekitar 12 menit |
| Loop pelatihan 35 epoch pada fitur beku | di bawah 60 detik |
| Satu permintaan web lengkap | 3,81 detik |

Tanpa feature caching, 200 epoch akan menjalankan backbone 414.000 kali. Dengan
cache, backbone dijalankan 2070 kali dan sisanya hanya dua lapisan Linear.

## 7. Keluaran pytest

```
343 passed, 1 warning in 146.48s
```

Satu-satunya peringatan adalah `StarletteDeprecationWarning` dari
`fastapi.testclient` yang menyarankan `httpx2`. Tidak ada tes yang gagal,
dilewati, atau dilewati bersyarat.

## 8. Kesimpulan

F1 makro 1,0000 pada data uji untuk kedua head tercapai pada dataset ini, dengan
catatan ukuran data uji yang kecil dan kepercayaan rata-rata yang sedang. Yang
belum tervalidasi adalah klasifikasi bentuk dari objek segmentasi, dan itu klaim yang tidak bisa ditutup dengan menukar parameter segmentasi.
