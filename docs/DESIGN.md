# DESIGN: Spesifikasi Antarmuka dan Desain Visual

## 1. Ruang Lingkup
Dokumen ini mengatur tampilan antarmuka web sederhana (C7 pada ARCHITECTURE.md). Bagian teknis modul Python (preprocess, model, label_map, train, evaluate, infer, visualize) tetap tercantum di bagian 8.

## 2. Palet Warna

| Peran | Nilai | Keterangan |
|-------|-------|------------|
| Primary | `#050505` | Teks utama, border solid 2px, judul bagian, aksen struktural |
| Secondary | `#333333` | Teks sekunder, label metadata |
| Surface | `#FFFFFF` | Latar kartu hasil, panel konten utama |
| Sub-surface | `#EAE8E3` | Latar kontrol tombol alih, pratinjau data |
| Background | `#F4F4F0` | Latar halaman utama (matte unbleached paper) |
| Accent | `#D31515` | Tombol proses utama, aksen peringatan, fokus aktif |
| Level Tinggi | `#005A36` | Teks level confidence Tinggi, 7,57:1 pada latar paper |
| Level Sedang | `#8A5000` | Teks level confidence Sedang, 5,90:1 pada latar paper |
| Level Rendah | `#B30000` | Teks level confidence Rendah, 6,53:1 pada latar paper |
| Muted | `#555555` | Teks sekunder redup, 6,76:1 pada latar paper |
| Latar Peringatan | `#FFF9E6` | Latar kotak peringatan confidence rendah |
| Latar Galat | `#FFF0F0` | Latar kotak pesan galat |

Teks utama menggunakan warna gelap (#050505). Warna teks pada latar `#D31515` wajib putih (`#ffffff`).

### 2.1 Aksesibilitas Warna
- Teks di atas `#D31515` harus putih (`#ffffff`).
- Informasi status tidak boleh bergantung pada warna saja. Setiap level confidence wajib disertai teks (Tinggi, Sedang, Rendah).
- Kontras teks terhadap latar wajib memenuhi WCAG 2.1 AA (rasio minimal 4,5:1). Rasio diuji pada tahap implementasi.

## 3. Tipografi
- Font struktural: sans-serif sistem (-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif).
- Font telemeteri/kode: monospaced (ui-monospace, Consolas, "Courier New", monospace).
- Judul halaman: 28 px, tebal (weight 800), sans, uppercase.
- Judul bagian: 18 px, tebal (weight 800), sans, uppercase.
- Subjudul / kategori: 14 px, tebal (weight 700), mono, uppercase.
- Teks isi: 16 px, normal (weight 400), sans, line-height 1.5.
- Teks keterangan: 13 px, medium (weight 500), mono, uppercase.
- Kode & metrik angka: 13 px, semibold (weight 600), mono, tabular-nums.

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
- Kanan: satu panel visualisasi segmentasi dengan ukuran tetap, dan satu baris tombol alih tahap di bawahnya untuk berpindah di antara lima tahap praproses, dari citra asli hingga tahap segmentasi. Panelnya tetap satu, bukan lima panel terpisah.

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
