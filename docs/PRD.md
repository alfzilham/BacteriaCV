# PRD: Prototipe Klasifikasi Bentuk Sel dan Status Gram

> STATUS: FINAL (versi 1.0). Keputusan produk sudah disepakati melalui sesi product-brainstorming.

## 1. Ringkasan
Prototipe Computer Vision yang menerima satu citra mikroskop digital dan menampilkan bentuk sel, status Gram, confidence score per head, dan visualisasi setiap tahap pra-pemrosesan secara berurutan. Dijalankan di laptop atau PC dengan GPU konsumen.

## 2. Masalah
Klasifikasi bakteri secara manual memakan waktu dan rentan terhadap kesalahan pembaca. Sementara itu, mahasiswa yang belajar integrasi mikrobiologi dan Computer Vision sulit menghubungkan teori pewarnaan Gram dengan proses komputasi yang sebenarnya. Prototipe ini menjembatani keduanya.

## 3. Tujuan Produk
1. Membantu mahasiswa memahami alur dari citra mikroskop hingga prediksi.
2. Menyediakan prototipe yang dapat dijalankan dan dimodifikasi.
3. Menunjukkan bahwa satu backbone CNN dapat memprediksi bentuk dan status Gram sekaligus.
4. Menjadi dasar yang dapat dikembangkan untuk dataset lain.

Prioritas bila terjadi konflik: kualitas laporan dan rubrik didahulukan; prototipe cukup berjalan untuk demo.

## 4. Pengguna

| Segmen | Peran | Kebutuhan utama |
|--------|-------|-----------------|
| Mahasiswa (utama) | Pengguna langsung | Memahami alur, memodifikasi kode, mendapat hasil cepat, membandingkan dengan label asli |
| Peneliti (sekunder) | Pengembang lanjutan | Struktur modul yang jelas dan titik ekstensi |
| Ahli mikrobiologi (sekunder) | Pengguna alat bantu | Hasil dengan confidence jelas dan peringatan bila tidak pasti |

### 4.1 Urutan Prioritas Kebutuhan Mahasiswa
1. Memahami alur (tertinggi): visualisasi setiap tahap pra-pemrosesan.
2. Memodifikasi kode: modul terstruktur sesuai DESIGN.md bagian 8.
3. Hasil cepat: tercakup oleh kriteria demo.
4. Membandingkan dengan label asli: fitur pengembangan, karena antarmuka demo tidak memuat data uji.

## 5. Cerita Pengguna
1. Sebagai mahasiswa, saya ingin melihat citra asli, hasil resize, normalisasi, segmentasi, dan hasil akhir secara berurutan, agar saya dapat mencocokkan langkah-langkah itu dengan teori di laporan.
2. Sebagai mahasiswa, saya ingin melihat keterangan khusus pada tahap yang gagal, agar saya tahu di mana masalah terjadi.
3. Sebagai mahasiswa, saya ingin mengunggah citra dan mendapat label bentuk serta status Gram beserta confidence-nya, agar saya bisa menggunakannya untuk tugas.
4. Sebagai mahasiswa, saya ingin membuka dan mengubah kode per modul, agar saya dapat bereksperimen.
5. Sebagai peneliti, saya ingin menambahkan dataset atau arsitektur baru melalui titik ekstensi yang didokumentasikan, agar pengembangan tidak mengubah seluruh sistem.
6. Sebagai ahli mikrobiologi, saya ingin melihat peringatan saat confidence rendah, agar saya tidak mengandalkan hasil yang tidak pasti.

## 6. Kebutuhan
Mengacu pada SPEC.md bagian 4, ditambah kebutuhan UX berikut:

| ID | Kebutuhan | Prioritas |
|----|-----------|-----------|
| U1 | Panel bernomor menampilkan setiap tahap pra-pemrosesan secara berurutan | Wajib |
| U2 | Tombol "lanjut" dan "kembali" untuk menelusuri tahap | Wajib |
| U3 | Tahap yang gagal ditandai dengan keterangan khusus dan alasan singkat | Wajib |
| U4 | Keterangan singkat untuk setiap tahap, sesuai teori di laporan | Wajib |
| U5 | Peringatan bila confidence di bawah 0,6 | Wajib |
| U6 | Grad-CAM atau visualisasi aktivasi lapisan | Tidak termasuk versi awal |
| U7 | Perbandingan hasil dengan label asli dari dataset | Pengembangan |

## 7. Kriteria Keberhasilan

### 7.1 Demo (batas minimal, disepakati)
Sistem dapat menerima satu citra dan menampilkan label bentuk, label Gram, confidence per head, dan panel tahap pra-pemrosesan.

### 7.2 Laporan
- Acuan utama: F1-score makro per head pada data uji, dari eksperimen nyata.
- Pelengkap: akurasi per head.
- Proyeksi 95% hingga 99% dari literatur adalah pembanding, bukan syarat.
- Non-fungsional: inferensi satu citra di GPU konsumen diukur dan dilaporkan; tidak ada ambang yang dipaksakan sebelum diukur.

### 7.3 Pengalaman Pengguna
- Mahasiswa dapat menjelaskan urutan tahap pra-pemrosesan setelah menggunakan prototipe, diuji melalui sesi singkat dengan minimal tiga mahasiswa (metode dan hasil perlu didokumentasikan).

## 8. Di Luar Lingkup
Sesuai SPEC.md bagian 2, ditambah:
- Visualisasi aktivasi lapisan CNN (U6).
- Perbandingan dengan label asli di antarmuka (U7).

## 9. Risiko

| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| Pembagian acak melebih-lebihkan performa | hasil tidak mencerminkan galur baru | dicatat sebagai keterbatasan, uji eksternal sebagai pengembangan |
| Prediksi per citra kehilangan citra campuran | label tidak lengkap | dicatat sebagai keterbatasan |
| Salah pakai sebagai diagnosis | risiko klinis | peringatan di antarmuka, batasan di SPEC |
| Watershed gagal pada citra padat | segmentasi buruk | tahap ditandai gagal, alur tetap ditampilkan (U3) |
| Panel tahap terlalu teknis untuk pemula | pemahaman rendah | keterangan singkat per tahap (U4) |

## 10. Keputusan yang Sudah Disepakati
| Keputusan | Sumber |
|-----------|--------|
| Prioritas konflik: laporan didahulukan | Sesi product-brainstorming |
| Batas demo: satu citra, label, confidence, visualisasi | Sesi product-brainstorming |
| Pengguna utama: mahasiswa | Sesi product-brainstorming |
| Urutan kebutuhan mahasiswa: alur, kode, hasil cepat, label asli | Sesi product-brainstorming |
| Tampilan tahap pra-pemrosesan berurutan, tanpa Grad-CAM | Sesi product-brainstorming |
| Tahap gagal ditampilkan dengan keterangan khusus | Sesi product-brainstorming |
