# Kriteria Sukses Segmentasi DIBaS

Ditetapkan sebelum eksperimen, 4 Oktober 2026, oleh pemilik proyek.

## Latar Belakang

Segmentasi morfologis adalah keputusan arsitektur A4 danentropy untuk output F5
di SPEC bagian 4. Parameternya belum tervalidasi, sehingga tidak boleh dipakai
sebagai klaim kapasitas sistem sebelum lolos pengukuran.

## Parameter yang Diuji

- `min_distance`: 8, 12, 16 (dimulai dari 12)
- Skala: resolusi asli 2048 x 1532
- Disk erosi dan dilasi: `disk(5)`, jejak 11 x 11
- `min_object_area`: 300
- Ambang: Otsu pada citra grayscale yang sudah diblur

## KLASIFIKASI YANG DIGUNAKAN

Menggunakan `label_map.LOOKUP` yang sudah tervalidasi audit, bukan penentuan ulang:

- Kokus: 9 spesies
- Batang: 23 spesies

## KRITERIA LANGSUNG BERHASIL

Pada `min_distance = 12`, ketiga syarat berikut harus terpenuhi:

1. Elongasi median seluruh spesies kokus tidak boleh lebih dari 1,45
2. Elongasi median seluruh spesies batang tidak boleh kurang dari 1,60
3. Jarak antara median kokus dan median batang minimal 0,15

Elongasi dihitung sebagai `axis_major_length / axis_minor_length` dari
`skimage.measure.regionprops` pada setiap objek hasil watershed._statistik
yang dipakai adalah median, bukan rata-rata, karena distributionnya tidak normal.

## KRITERIA GAGAL

Eksperimen dinyatakan gagal bila setelah `min_distance` diuji pada 8, 12, dan
16, ada `min_distance` yang memenuhi ketiga syarat di atas.

## BATAS WAKTU

30 menit. Bila habis tanpa memenuhi kriteria, eksperimen dihentikan.

## KEPUTUSAN YANG MENGIKUTI

### Bila berhasil

Opsi (c) diteruskan. Segmentasi berjalan pada resolusi asli untuk mask dan
visualisasi, resize 224 hanya untuk tensor model. Konstan di `config.py`
diperbarui ke nilai yang terpilih.

### Bila gagal

Segmenasi TIDAK dibuang. Fallback adalah opsi (c) yang diperbaiki dengan label
akurat:

1. Istilah "kontur sel" di ARCHITECTURE C6 dan F5 di SPEC bagian 4 diubah
   menjadi "kontur area bakteri", karena yang tersegmentasi adalah kelompok
   bakteri, bukan sel individual. Ini amendemen terpisah.
2. SPEC bagian 8 mencatat keterbatasan dengan angka terukur: segmentasi
   memisahkan kelompok bakteri tetapi bukan sel individual, karena sel
   bersentuhan pada citra Gram 100x.
3. Tabel elongasi per spesies masuk laporan.

Alasan fallback dipilih, bukan pembuangan segmentasi: F5 berstatus Wajib di
SPEC bagian 4, dan ARCHITECTURE A3 menyatakan segmentasi tidak dibutuhkan pada
inferensi sehingga segmentasi tidak memengaruhi akurasi model. Segmentasi yang
kurang sempurna dengan label yang benar masih bernilai; yang merusak adalah
label yang salah.

## CATATAN TENTANG STAPHYLOCOCCUS

Klaster anggur pada Staphylococcus adalah unit morfologis yang nyata pada
perbesaran 100x, bukan artefak segmentasi. Objek yang lebih besar pada spesies
kokus klaster tidak otomatis berarti segmentasi gagal.