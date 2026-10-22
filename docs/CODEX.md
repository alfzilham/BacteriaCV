# CODEX.md - Cara kerja BacteriaCVMobile dengan satu sesi

Tanggal   : 12 Oktober 2026
Untuk     : Codex CLI, satu sesi
Status    : dokumen ini menggantikan pemisahan beberapa sesi agent

---

## 1. FOKUS

Repo ini adalah **BacteriaCVMobile**, aplikasi Android yang mengklasifikasi
bentuk sel dan status Gram bakteri sepenuhnya di perangkat.

Repo **BacteriaCV** adalah aplikasi web. Consider it selesai. Jangan
bekerja di sana kecuali diminta secara eksplisit.

Semua perintah dan aturan di bawah berlaku untuk BacteriaCVMobile.

---

## 2. DUA PERAN DALAM SATU SESI

Dulu ada dua sesi terpisah: satu menulis kode, satu mengaudit. Sekarang
hanya ada satu.

```
SEBELUM   penulis  ->  auditor  ->  PR  ->  owner merge

SEKARANG  Codex menulis
          Codex memverifikasi karyanya sendiri
          Codex membuat PR
          Owner merge
```

Ini bukan independen. Jangan perlakukan seolah-oke. Baca aturan self-audit
di bagian 4 sebelum memverifikasi apa pun.

Yang **tidak berubah**: Codex tidak pernah merge. Owner yang merge.

---

## 3. URUTAN LANGKAH

Ikuti urutan ini untuk setiap task. Jangan melompat.

```
1. Baca AGENTS.md di repo ini, sampai bagian Batas keras
2. Baca docs/CHECKPOINT.md untuk tahu di mana pekerjaan berhenti
3. Jalankan git status SEBELUM menyentuh apa pun
4. Kalau ada perubahan belum commit dari sesi lain:
   STOP dan lapor. Jangan menimpa, jangan pop stash
5. Kerjakan SATU hal saja
6. Verifikasi hitung ulang, lihat bagian 4
7. Branch, bukan master
8. Commit dengan pesan yang menyebut apa yang berubah
9. Push ke branch itu
10. Buka pull request
11. Laporkan, lalu berhenti. Jangan merge
```

---

## 4. ATURAN SELF-AUDIT

Ini yang paling penting. Semua aturan di sini berlaku setiap kali Codex
memverifikasi pekerjaan sendiri.

### Wajib

```
1. Hitung ulang angka dari sumber. Jangan mengambil angka dari output
   yang sama yang menghasilkan angka itu.

2. Kalau sebuah tes menghasilkan angka, konfirmasi lewat jalur lain
   sebelum menyebutnya benar.

3. Setiap klaim yang berasal dari laporan sendiri diberi label
   "belum diverifikasi independen". Jangan tulis seolah sudah pasti.

4. Kalau menemukan masalah pada pekerjaan sendiri, WAJIB laporkan.
   Diam di sini akan mengulang kesalahan sendiri tanpa ada yang
   memberitahu.

5. Angka berikut dihitung dari sumber SETIAP KALI, bukan dari catatan
   sebelumnya:
   - SHA-256 heads.pt
   - urutan GRAM_LABELS di config.py
   - F1 kedua head
   - jumlah tes yang lulus
```

### Yang tidak boleh

```
- Menyatakan LULUS karena tes hijau, tanpa memeriksa apakah tes itu
  benar-benar menguji apa yang diklaim
- Menghapus atau mengubah tes supaya bisa lulus
- Melaporkan angka yang tidak bisa ditelusuri ke perintah yang
  menghasilkannya
- Mengarang penyebab ketika penyebabnya belum diketahui

Kegagalan yang nyata lebih berharga daripada laporan yang bersih.
```

---

## 5. BATAS KERAS

Sudah dikunci dan tidak bisa ditawar. Semuanya proven working.

```
model            assets/spike/model_int8.onnx, 30,45 MB
                 BUKAN fp32, bukan int8 polos
provider         XNNPACK saja
                 NNAPI diukur 2.011 ms, tidak dipakai
ambang level     high >= 0,8, medium >= 0,6, low < 0,6
                 sama dengan aplikasi web, dikunci test
confidence       TIDAK ditampilkan sebagai persen
                 int8 meleset 0,542656 terhadap acuan float32
                 yang ditampilkan hanya level
disclaimer       keempat kalimat wajib ada
                 kalimat pertama untuk Play Store harus persis
package id       alfzilham.ilham.bacteriacv_mobile
                 tidak bisa diubah setelah rilis pertama
area_resize.dart 146 baris, presisi terhadap cv2
                 JANGAN pernah ditulis ulang
                 resize INTER_AREA tidak bisa dipindah ke dalam graf
```

Tidak ada server, tidak ada backend, tidak ada permission INTERNET,
tidak ada telemetry, tidak ada login. Semua inferensi di perangkat.

---

## 6. ALUR PULL REQUEST

```
Branch dari main, jangan pernah di master
Commits pushed ke branch itu
Pull request ke master
Owner merge

Tidak ada branch protection di repo ini. Repo privat di akun gratis,
GitHub menjawab HTTP 403. Jadi aturan ini hanya konvensi, dan pematuhan
nya bergantung pada Codex.
```

---

## 7. APA YANG SEHARUSNYA BELUM ADA

Known gaps yang diterima. Bukan defect.

```
kontras kotak disclaimer  1,22:1 terang, 1,51:1 gelap
                          border 3,48:1, jadi batasnya terlihat
                          sama seperti track scrollbar web 1,11:1
pratinjau decode dua kali 155-169 ms, perbaikannya menyentuh pipeline
warna Material 3         bukan palet DESIGN.md, keputusan produk
segmentasi on-device      Spike P4 menyimpulkan tidak layak
dua proses pytest         hanya probe test_build_index.py yang
                          terbukti, bukan seluruh suite
```

Kalau salah satu di atas muncul sebagai temuan baru, periksa dulu apakah
ini known gap sebelum melaporkannya sebagai defect.

---

## 8. VERIFIKASI

Perintah yang bisa dijalankan ulang. Jalankan, jangan menyalin angka
dari sini.

```powershell
cd D:\2026\Workspace\PersonalApps\.MobileAppProjects\BacteriaCV

git status --porcelain
git log --oneline -3

flutter analyze --fatal-infos
flutter test
```

Untuk Akurasi di perangkat:

```
flutter build apk --release
adb install -r build/app/outputs/flutter-apk/app-release.apk
```

Sebelum mengukur apa pun di perangkat, bersihkan data aplikasi:

```powershell
adb shell pm clear alfzilham.ilham.bacteriacv_mobile
```

Wajib, karena `createSessionFromAsset` menyalin asset ke cache dan tidak
menimpa berkas yang sudah ada. Model yang salah akan dipakai tanpa
terdeteksi kalau cache tidak dibersihkan. Ini ditemukan di Task T setelah
dua hari pengukuran salah.

Kalau nama asset diubah, nama yang BARU juga wajib dipakai, bukan hanya
membersihkan cache.

---

## 9. ANGKA YANG HARUS SELALU DIHITUNG DARI SUMBER

Jangan percaya catatan. Hitung ulang.

```
SHA-256 heads.pt          46F194B368603C3A, 35.017 byte
                          TIDAK PERNAH BERUBAH

GRAM_LABELS               ("positive", "negative")
                          urutan mengunci indeks kelas head B

F1 uji                    1,0000 kedua head, 66 dari 66
                          confidence shape 0,9812 / gram 0,9671
confusion shape           [[19,0],[0,47]]
confusion gram            [[47,0],[0,19]]

pytest web                381 passed
pytest mobile             56 passed
```

---

## 10. KESALAHAN YANG SUDAH TERCA DIATURAN INI

Semua ditemukan karena agent menolak laporan tanpa bukti, bukan karena
diberi tahu. Jangan ulangi.

```
1  gerbang CI dipasang dengan prefiks "CI / "
   nama check sebenarnya tanpa prefiks, merge mustahil

2  required_approving_review_count 1 dengan satu orang write access
   GitHub tidak izinkan approve PR sendiri, merge mustahil

3  hipotesis dua fixture saling menghapus folder
   sebenarnya dua PROSES yang bertabrakan
   dua perbaikan yang diusulkan tidak memperbaiki apa pun

4  prefiks \\?\ diklaim muncul karena path hilang
   diuji, tidak terjadi. Eliminasinya tidak valid

5  jumlah parameter di brief 23.569.401 dan 16.384
   benar 23.516.228 dan 8.196

6  "web sudah punya disclaimer medical device"
   web cuma punya 1 dari 4, mobile punya keempatnya

7  ambang kontras scrollbar 4,5:1
   WCAG 1.4.11 = 3:1 untuk elemen non-teks

8  213 ms per citra di mobile
   sebenarnya 248-263 ms, resize 32 ms bukan 7 ms

9  push langsung ke master tanpa PR, di repo tanpa proteksi
```

Pola yang berlaku: temuan yang dibaca langsung dari berkas hampir selalu
benar. Temuan yang diingat dari sesi sebelumnya sering salah.

Aturan yang menyertainta: baca dokumen yang sudah menjawab pertanyaan,
sebelum menulis jawaban sendiri. Tiga kesalahan pertama di daftar itu
semua karena dokumen yang benar sudah ada dan tidak dibaca.

---

## 11. YANG HARUS DILAPORKAN

Laporan akhir memuat minimal:

```
apa yang dikerjakan, per berkas
angka hasil, dengan sumber perhitungannya
apa yang tidak selesai, dan kenapa
```

Kalau ada klaim yang tidak bisa diverifikasi independen, tulis begitu.
Pemilik proyek lebih butuh kejujuran daripada kelengkapan.