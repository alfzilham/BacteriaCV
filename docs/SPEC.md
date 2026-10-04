# SPEC: Sistem Klasifikasi Bentuk Sel dan Status Gram Bakteri

## 1. Tujuan
Sistem Computer Vision yang memprediksi bentuk sel (cocci, bacilli, spiral) dan status Gram (positif, negatif) bakteri secara simultan dari satu citra mikroskop digital, menggunakan satu backbone ResNet-50 dengan dua classification head.

## 2. Ruang Lingkup
### Termasuk versi awal
- Pra-pemrosesan citra: resize, normalisasi, segmentasi morfologis, transformasi watershed, augmentasi.
- Pelatihan dua head dengan backbone dibekukan.
- Lookup table pemetaan 32 spesies ke pasangan (bentuk, status Gram) untuk label pelatihan. Candida albicans dikeluarkan dari pelatihan.
- Evaluasi F1-score makro (utama) dan akurasi (pelengkap) per head.
- Antarmuka web sederhana untuk unggah citra dan melihat hasil.

### Tidak termasuk versi awal
- Fine-tuning penuh backbone (jalur pengembangan lanjutan).
- Klasifikasi per sel (jalur pengembangan lanjutan).
- Uji generalisasi pada galur bakteri baru.
- Integrasi ke sistem laboratorium klinis.

## 3. Data
- Dataset: DIBaS (Digital Images of Bacterial Species), 692 berkas citra, 33 spesies, resolusi asli 2048 × 1532 piksel.
- Jumlah berkas 692 hasil verifikasi langsung terhadap arsip. Angka 660 yang disebut paper asli dan makalah lain tidak sesuai isi arsip: jumlah citra per spesies tidak seragam, 21 spesies punya 20 berkas, 10 spesies punya 23, dan Veillonella punya 22.
- Tiga berkas tidak dapat dibaca oleh pustaka citra dan sudah rusak di dalam arsip DIBaS: Listeria monocytogenes 0023 berukuran 0 byte, serta Micrococcus spp 0021 dan 0023 dengan struktur TIFF rusak. SHA-256 berkas di disk identik dengan entri arsip, jadi kerusakan berasal dari sumber. Daftar lengkap beserta alasannya ada di `data/raw/unreadable.csv`.
- Spesies yang dilatih: 32 dari 33. Candida albicans dikeluarkan karena merupakan jamur, bukan bakteri, dan karena sel jamur berukuran 5 sampai 10 mikron sehingga kelas ini akan terpisah mudah oleh backbone dan menaikkan F1-score tanpa menunjukkan kemampuan klasifikasi morfologi bakteri. Pengecualian dicatat pada `bacteriacv/datasets/species_map.py`.
- Data efektif untuk pelatihan: 669 citra terbaca dari 32 spesies.
- Pembagian: acak per citra, 70% latih, 20% validasi, 10% uji, dengan validasi silang lima lipatan.
- Hasil pembagian nyata: 467 latih, 136 validasi, 66 uji.
- Bentuk sel: hanya dua kelas terisi, yaitu cocci dan bacilli. DIBaS tidak memuat spesies berbentuk spiral, sehingga kelas spiral dicatat sebagai tidak terisi.
- Lima lipatan hanya dibagikan di dalam data latih. Data validasi dipakai untuk early stopping, data uji hanya dipakai sekali di akhir.
- Lipatan diseimbangkan per spesies, bukan secara global, sehingga ukuran absolut tiap lipatan tidak sama (104/96/96/96/75). Stratifikasi tetap terjaga karena setiap spesies muncul di setiap lipatan.
- Pembagian acak tidak mengukur generalisasi ke galur baru. Hasil uji hanya merepresentasikan performa pada galur yang mirip dengan data latih.
- Label pelatihan diturunkan dari label spesies melalui lookup table.
- Citra dengan label yang tidak konsisten terhadap lookup table dibuang dari pelatihan.
- Citra yang tidak dapat dibaca oleh pustaka citra dibuang dari index dan dicatat di `data/raw/unreadable.csv` beserta alasannya.

## 4. Fungsi Sistem
| ID | Kebutuhan | Prioritas |
|----|-----------|-----------|
| F1 | Menerima citra mikroskop dalam format umum (PNG, JPG, TIFF) | Wajib |
| F2 | Melakukan pra-pemrosesan sesuai spesifikasi | Wajib |
| F3 | Menghasilkan prediksi bentuk sel dengan confidence score | Wajib |
| F4 | Menghasilkan prediksi status Gram dengan confidence score | Wajib |
| F5 | Menampilkan visualisasi hasil segmentasi | Wajib |
| F6 | Melatih model dengan class weighting dari data latih | Wajib |
| F7 | Menghitung F1-score makro dan akurasi per head | Wajib |
| F8 | Antarmuka web untuk unggah dan tampilan hasil | Wajib |

## 5. Keluaran
- Label bentuk sel dan confidence score.
- Label status Gram dan confidence score.
- Visualisasi hasil segmentasi.

## 6. Metrik Keberhasilan
- Acuan utama: F1-score makro per head.
- Pelengkap: akurasi per head.
- Rujukan pembanding: Talo (2019) 99,2%, Mai & Ishibashi (2021) 98,25%. Perbandingan langsung hanya pada metrik akurasi.
- Proyeksi laporan: kisaran 95% hingga 99%. Angka ini bukan syarat keberhasilan; hasil eksperimen nyata yang dilaporkan.

## 7. Penanganan Ketidakseimbangan Kelas
- Class weighting pada loss function.
- Bobot dihitung hanya dari data latih, bukan data validasi atau uji.

## 8. Batasan dan Keterbatasan
1. Prediksi per citra menghasilkan satu label dominan, sehingga citra dengan campuran bentuk sel tidak terwakili sepenuhnya.
2. Pembagian acak tidak mengukur generalisasi ke galur baru.
3. Performa bergantung pada kualitas citra dan konsistensi protokol pewarnaan Gram.
4. Sistem adalah alat bantu diagnostik awal, bukan pengganti kultur dan uji biokimia.

## 9. Kriteria Penerimaan

### 9.1 Kriteria Demo (wajib, batas minimal)
- Sistem menerima satu citra mikroskop dalam format PNG, JPG, atau TIFF.
- Sistem menampilkan label bentuk sel dan confidence score.
- Sistem menampilkan label status Gram dan confidence score.
- Sistem menampilkan visualisasi hasil segmentasi.
- Antarmuka web dapat dijalankan di laptop atau PC dengan GPU konsumen.

### 9.2 Kriteria Laporan (wajib untuk rubrik)
- F1-score makro dan akurasi dihitung per head pada data uji, dari eksperimen nyata.
- Angka yang dilaporkan adalah hasil eksperimen, termasuk jika lebih rendah dari proyeksi laporan.
- Keterbatasan pada Bagian 8 tercantum di setiap laporan hasil.
- Tidak ada klaim generalisasi ke galur baru.

### 9.3 Catatan
Pipeline penuh pada 669 citra menjadi target pengembangan, bukan syarat demo.
