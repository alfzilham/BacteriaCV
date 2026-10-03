# DESIGN: Spesifikasi Antarmuka dan Desain Visual

## 1. Ruang Lingkup
Dokumen ini mengatur tampilan antarmuka web sederhana (C7 pada ARCHITECTURE.md). Bagian teknis modul Python (preprocess, model, label_map, train, evaluate, infer, visualize) tetap tercantum di bagian 8.

## 2. Palet Warna

| Peran | Nilai | Keterangan |
|-------|-------|------------|
| Primary | `#588157` | Tombol utama, judul bagian, tautan |
| Secondary | `#A3B18A` | Latar panel sekunder, tombol sekunder, indikator confidence sedang |
| Surface | `#DAD7CD` | Latar kartu hasil, panel samping |
| Background | `#ffffff` | Latar halaman utama |

Teks utama menggunakan warna gelap dari palet di atas. Warna teks pada latar `#588157` wajib putih (`#ffffff`).

### 2.1 Aksesibilitas Warna
- Teks di atas `#588157` harus putih (`#ffffff`).
- Informasi status tidak boleh bergantung pada warna saja. Setiap level confidence wajib disertai teks (Tinggi, Sedang, Rendah).
- Kontras teks terhadap latar wajib memenuhi WCAG 2.1 AA (rasio minimal 4,5:1). Rasio diuji pada tahap implementasi.

## 3. Tipografi
- Font: sans-serif sistem (Arial sebagai cadangan).
- Judul halaman: 24 px, tebal.
- Judul bagian: 18 px, tebal.
- Teks isi: 16 px.
- Teks keterangan: 13 px.

## 4. Layout

Layout mengikuti skeleton yang disepakati, yaitu satu halaman dengan tiga area utama.

```
+--------------------------------------------------------------+
| [Logo]   Menu       Menu     [Tombol sekunder] [Tombol utama]|  Header
+--------------------------------------------------------------+
| +-------------------------+  +--------------------------+     |
| | Kartu unggah citra      |  | Kartu ringkasan hasil    |     |
| | [area unggah]           |  | [label + confidence]     |     |  Baris atas
| +-------------------------+  +--------------------------+     |
|                                                             |
| Judul bagian                | +----------------------------+  |
| Teks keterangan             | |                            |  |
| [Tombol proses] [Pilihan]   | |   Visualisasi segmentasi   |  |  Area hasil
| Ringkasan teks hasil        | |                            |  |
|                             | +----------------------------+  |
|                             | Panel kontrol: opsi tampilan   |
+--------------------------------------------------------------+
| [Chip] [Chip] [Chip] [Chip] [Chip]                           |  Footer: keterangan
+--------------------------------------------------------------+
```

### 4.1 Header
- Logo atau nama sistem di kiri.
- Menu navigasi di tengah: Unggah, Riwayat, Tentang.
- Tombol sekunder dan tombol utama di kanan.

### 4.2 Baris Atas
- Kiri: kartu unggah citra dengan area drag-and-drop dan tombol pilih berkas.
- Kanan: kartu ringkasan hasil dengan label dan confidence untuk Head A dan Head B.

### 4.3 Area Utama
- Kiri: judul bagian, teks keterangan, tombol proses, dan teks hasil ringkas.
- Kanan: panel visualisasi segmentasi dengan ukuran tetap, dan panel kontrol di bawahnya dengan toggle untuk menampilkan atau menyembunyikan kontur.

### 4.4 Footer
- Lima chip informasi: format yang didukung, ukuran maksimal, versi model, sumber data, dan disclaimer.

## 5. Komponen

| Komponen | Keadaan | Perilaku |
|----------|---------|----------|
| Area unggah | normal, hover, dragover, error | menerima PNG, JPG, TIFF hingga 20 MB |
| Tombol proses | normal, disabled (belum ada citra), loading | nonaktif sebelum citra dipilih |
| Kartu hasil | kosong, memuat, terisi | menampilkan label dan confidence per head |
| Label confidence | tinggi (>= 0,8), sedang (0,6 hingga 0,79), rendah (< 0,6) | rendah menampilkan peringatan |
| Panel visualisasi | kosong, memuat, terisi | gambar overlay kontur, toggle untuk sembunyikan |
| Peringatan | tampil saat confidence rendah | teks: "Hasil ini sebagai alat bantu, bukan diagnosis." |
| Chip footer | statis | informasi, tidak interaktif |

## 6. Interaksi
1. Pengguna mengunggah citra melalui area unggah atau drag-and-drop.
2. Pratinjau citra tampil di kartu unggah.
3. Pengguna menekan tombol proses. Tombol berubah menjadi keadaan memuat.
4. Kartu hasil dan panel visualisasi terisi bersamaan.
5. Bila confidence rendah, peringatan tampil di atas kartu hasil.
6. Pengguna dapat menyembunyikan kontur dengan toggle.

## 7. Pesan Kesalahan
| Kondisi | Pesan |
|---------|-------|
| Format tidak didukung | "Format berkas tidak didukung. Gunakan PNG, JPG, atau TIFF." |
| Ukuran melebihi 20 MB | "Ukuran berkas melebihi 20 MB." |
| Citra tidak terbaca | "Citra tidak dapat dibaca. Coba unggah berkas lain." |
| Segmentasi gagal | "Segmentasi tidak berhasil. Hasil klasifikasi tetap ditampilkan tanpa visualisasi." |

## 8. Bagian Teknis Modul
Modul Python dan kontrak fungsinya tetap sesuai dengan DESIGN.md versi sebelumnya:
- `preprocess.py`: `load_image`, `segment_cells`, `preprocess`.
- `model.py`: `BacteriaNet` dengan dua head.
- `label_map.py`: `LOOKUP`, `to_targets`, `class_weights`.
- `train.py`: loss berbobot, optimizer Adam, early stopping pada F1 makro validasi.
- `evaluate.py`: `evaluate` mengembalikan F1 makro dan akurasi per head.
- `infer.py`: `predict` mengembalikan label, confidence, dan overlay.
- `app/main.py`: `POST /predict`, `GET /`, batas ukuran 20 MB.

## 9. Pengujian Antarmuka
| Jenis | Cakupan |
|-------|---------|
| Visual | kesesuaian dengan layout bagian 4 dan palet bagian 2 |
| Keadaan | setiap keadaan pada bagian 5 diuji secara terpisah |
| Aksesibilitas | kontras teks dan keterbatasan warna pada bagian 2.1 |
| Unggah | berkas valid, berkas tidak didukung, dan berkas melebihi batas |
