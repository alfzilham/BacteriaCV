# CONTEXT: Latar Belakang Proyek

> STATUS: FINAL (versi 1.0). Pengguna dan tujuan sudah divalidasi melalui sesi product-brainstorming.

## 1. Asal Proyek
Proyek ini merupakan implementasi dari studi kasus A pada tugas mata kuliah Biologi, dengan judul laporan "Desain Konseptual Sistem Komputasi Berbasis Computer Vision untuk Klasifikasi Bentuk Sel dan Status Gram Bakteri dari Citra Mikroskopis Digital". Laporan ditulis oleh Alfiz Ilham (260411101100105), Program Studi Teknik Komputer, Universitas Syiah Kuala, 2026.

## 2. Masalah
Identifikasi bakteri melalui pewarnaan Gram bergantung pada keterampilan pembaca. Samuel dkk. (2016) mencatat tingkat kesalahan 0,4% hingga 2,7%, dengan 24% kesalahan berasal dari pembaca.

Di sisi pendidikan, mahasiswa yang belajar integrasi mikrobiologi dan Computer Vision sering kesulitan menghubungkan teori pewarnaan dan morfologi dengan proses komputasi yang sebenarnya.

## 3. Pengguna
- **Mahasiswa (utama):** mempelajari integrasi mikrobiologi dan Computer Vision, dan memodifikasi kode untuk tugas.
- **Peneliti (sekunder):** ingin mengembangkan arsitektur multi-head pada dataset lain.
- **Ahli mikrobiologi (sekunder):** calon pengguna alat bantu diagnostik awal.

Detail kebutuhan ada di PRD.md bagian 4 dan 5.

## 4. Tujuan dan Prioritas
Tujuan proyek bersifat gabungan: lulus tugas dengan kualitas tinggi, membuktikan konsep, membangun prototipe yang dapat dikembangkan, dan memahami integrasi mikrobiologi dengan Computer Vision.

Prioritas bila terjadi konflik: kualitas laporan dan rubrik didahulukan; prototipe cukup berjalan untuk demo.

## 5. Di Luar Lingkup
- Penggunaan klinis.
- Identifikasi spesies (hanya bentuk dan status Gram).
- Pengganti kultur dan uji biokimia.

## 6. Istilah
| Istilah | Arti |
|---------|------|
| Cocci | sel berbentuk bulat |
| Bacilli | sel berbentuk batang |
| Spiral | sel berbentuk heliks atau melengkung |
| Gram-positif | dinding sel tebal dengan peptidoglikan, tampak ungu |
| Gram-negatif | dinding sel tipis dengan membran luar, tampak merah muda |
| DIBaS | dataset publik 692 citra dari 33 spesies; 672 citra dari 32 spesies dipakai setelah Candida albicans dikeluarkan |
| Lookup table | pemetaan label spesies ke pasangan (bentuk, Gram) |
| Multi-head | satu backbone dengan beberapa classification head |

## 7. Sumber Utama
- Talo, M. (2019). arXiv:1912.08765
- Mai, D.-T., & Ishibashi, K. (2021). Electronics, 10(23), 3005
- Cabeen, M. T., & Jacobs-Wagner, C. (2005). Nature Reviews Microbiology, 3(8), 601-610
- Samuel, L. P., dkk. (2016). Journal of Clinical Microbiology, 54(6), 1442-1447
- Zieliński, B., dkk. (2017). PLoS ONE, 12(9), e0184554
- Daftar lengkap ada di laporan, Daftar Pustaka.

## 8. Keputusan yang Sudah Disepakati
Lihat SPEC.md bagian 2 sampai 9 dan PRD.md bagian 10. Ringkasan: pembagian acak per citra, F1-score makro sebagai acuan, class weighting, klasifikasi per citra, antarmuka web sederhana dengan panel tahap pra-pemrosesan berurutan.
