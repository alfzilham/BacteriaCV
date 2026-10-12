# CHECKPOINT - BacteriaCVMobile

Tanggal checkpoint : 12 Oktober 2026
Untuk              : Codex CLI, satu sesi
Status             : mobile berjalan, tidak ada blocker teknis

> **FOKUS PINDAI MOBILE.** Repo `BacteriaCV` dianggap selesai oleh owner
> dan tidak perlu dikerjakan lagi kecuali diminta. Panduan lengkap untuk
> sesi Codex ada di `docs/CODEX.md`. Baca itu lebih dulu.

---

## 1. PERAN

```
Codex CLI    penulis sekaligus auditor, satu sesi
Owner        membuat keputusan, merge pull request

Codex tidak pernah merge. Owner merge.
```

Karena penulis dan auditor adalah orang yang sama, ia bukan auditor
independen. Aturan self-audit ada di `docs/CODEX.md` bagian 4 dan di
`AGENTS.md` mobile. Wajib dipatuhi setiap kali memverifikasi pekerjaan
sendiri.

---

## 2. LOKASI

```
BacteriaCV         D:\2026\Workspace\PersonalApps\BacteriaCV
BacteriaCVMobile   D:\2026\Workspace\PersonalApps\.MobileAppProjects\BacteriaCV
File instruksi    D:\2026\Workspace\PersonalApps\BacteriaCV\docs\prompts\
Perangkat         R9RYC0839YH, Samsung SM-A075F, Helio G99, SDK 36
```

Prompt untuk agent ada di `docs\prompts\`, satu berkas per agent:

```
BCV-2_UI_PROMPTS.md
BCV-3_Docs_PROMPTS.md
BCV-4_Mobile_PROMPTS.md
```

Isi di dalam code box. Di luar code box tidak boleh disentuh.

---

## 3. ANGKA YANG SUDAH TERVERIFIKASI

Semua di bawah ini dihitung ulang, bukan diambil dari laporan.

```
DATASET        669 citra DIBaS, 32 spesies
               467 latih / 136 validasi / 66 uji

MODEL          ResNet50 beku, ImageNet V2, 2048 fitur
               head A Linear(2048,2) softmax
               head B Linear(2048,2) sigmoid
               23.516.228 parameter
               heads.pt 35.017 byte, SHA-256 46F194B368603C3A
               TIDAK PERNAH BERUBAH

METRIK UJI     F1 makro 1,0000 kedua head
               accuracy 1,0000 kedua head
               confidence shape 0,9812 / gram 0,9671
               confusion shape [[19,0],[0,47]]
               confusion gram  [[47,0],[0,19]]

PYTEST WEB     381 passed  (378 + 3 test pengaman)

PYTEST MOBILE  56 passed  (dari 37)

MOBILE int8    model 30,45 MB, F1 1,000000 lewat classifier.dart
               66 dari 66 benar, nol NaN
               XNNPACK 72 ms warm, total 256-333 ms
               unduhan flagship 47,1 MB
```

---

## 4. STATUS BACTERIAV WEB - SELESAI

Repo ini dianggap selesai oleh owner. Jangan dikerjakan kecuali diminta.

```
PR #1  ci matrix tiga versi Python          MERGED
PR #2  dokumentasi alur pull request       MERGED
PR #3  privacy policy + dokumen Play Store  MERGED
PR #4  GitHub Pages untuk privacy policy   MERGED
PR #5  isolasi folder temp per proses      MERGED

origin/main    c291d69
deploy         Railway hijau, HTTP 200
               /api/health -> {"status":"ready","checkpoint":"heads.pt"}
pytest         381 passed
```

Privacy policy **live**:

```
https://alfzilham.github.io/BacteriaCV/
```

URL itu root, bukan `/privacy/`. Workflow meng-upload isi folder sebagai
root, jadi privacy policy menjadi halaman depan. Tidak masalah untuk Play
Store, yang dibutuhkan hanya URL publik.

Branch protection aktif di `main`: push ditolak, tiga gerbang CI wajib.
Tidak bisa dipasang di BacteriaCVMobile karena repo privat di akun gratis,
GitHub menjawab HTTP 403. Jadi alur pull request di mobile hanya konvensi.

---

## 5. STATUS BACTERIACVMOBILE

```
PR #1  spike Task S, T, U                MERGED
PR #2  CI tiga job                        MERGED

CI        analyze, test, android build  hijau
```

Aplikasi sudah **proven working**:

```
dialog file picker   DocumentsUI terbuka, TIFF terlihat, dipilih
                     Analyze hidup, hasil 292 ms
                     screenshot di spike/artifacts/shots/
F1                    head A 1,000000, head B 1,000000, 66/66
```

Yang sudah dikerjakan:

```
32c42db  feat: turn the spike into an app people can use
7f978fb  test: prove the app is usable end to end
```

File aplikasi sekarang:

```
lib/main.dart                 12 baris
lib/app.dart                  42
lib/pipeline/classifier.dart  264
lib/ui/upload_screen.dart     212
lib/ui/result_screen.dart     126
lib/ui/disclaimer.dart        86
lib/spike/area_resize.dart    146   <- tidak pernah diubah sejak scaffold
```

Decision yang sudah dikunci di mobile:

```
model int8 30,45 MB, BUKAN fp32 103,4 MB
provider   XNNPACK, BUKAN NNAPI (2.011 ms)
confidence TIDAK ditampilkan sebagai persen, hanya level high/medium/low
ambang    0,8 dan 0,6, sama dengan web, dikunci test
```

---

## 6. SEDANG BERJALAN - TASK Z2 MOBILE

BCV-4 Mobile sedang mengerjakan tata letak di layar kecil dan font besar.

```
branch  layout-small-screens
status  M lib/ui/disclaimer.dart
        M lib/ui/result_screen.dart
       ?? test/layout_test.dart
       ?? test/_tmp_dump_test.dart
belum di-commit
```

File `_tmp_dump_test.dart` tidak jelas asalnya. Perlu diperiksa apakah itu
skrip sekali pakai yang seharusnya dihapus sebelum commit.

Pengaturan: JANGAN push ke master. Branch, push ke branch, owner buat PR.

---

## 7. TUGAS BELUM SELESAI

### Mobile

```
Task Z2 layout        sedang berjalan
build AAB             belum pernah, dan model tidak ada di git
                      butuh model di-commit dengan git add -f
                      atau disalin saat build
```

### Play Store

```
developer account     $25 sekali, belum dibuat
upload AAB            belum
Data Safety form      docs/DATA-SAFETY.md sudah ada, 3 placeholder
Health Apps form      docs/HEALTH-APPS.md sudah ada
listing deskripsi     belum, kalimat wajib sudah ditentukan
```

Kalimat wajib, persis, di paragraf pertama deskripsi:

```
This app is not a medical device and does not diagnose, treat, or
prevent any condition.
```

Privacy Policy URL untuk form Play:

```
https://alfzilham.github.io/BacteriaCV/
```

### Yang ditunda dengan sengaja

```
spec BCV-5 Triage    ditunda, nol temuan yang bisa diuji
kontras 1,22:1       kotak disclaimer mobile, biarkan
                     sama seperti track scrollbar web 1,11:1
pratinjau decode 2x  tradeoff diterima, 155-169 ms
warna Material 3     bukan palet DESIGN.md, keputusan produk
```

---

## 8. ANGKA YANG HARUS DIPERHATIKAN

### Kesalahan yang sudah tercatat

Semua ditemukan karena agent menolak laporan tanpa bukti, bukan karena
saya menyuruh. Jangan ulangi.

```
1  gerbang CI dipasang "CI / Tes pada Python 3.11"
   nama check sebenarnya tanpa prefiks "CI / "
   -> gerbang tidak pernah cocok, merge mustahil

2  required_approving_review_count: 1
   hanya ada satu orang dengan write access
   -> GitHub tidak izinkan approve PR sendiri, merge mustahil

3  hipotesis "dua fixture saling menghapus folder"
   sebenarnya dua PROSES pytest yang bertabrakan
   -> dua pilihan perbaikan yang saya usulkan tidak memperbaiki apa pun

4  klaim prefiks \\?\ muncul karena path hilang
   diuji, resolve() pada file terhapus tidak menghasilkan prefiks itu
   -> eliminasinya tidak valid, belum diketahui cause sebenarnya
5  jumlah parameter di brief mobile 23.569.401 dan 16.384
   benar 23.516.228 dan 8.196
   -> tidak masuk dokumen terbit, tidak perlu ditarik

6  "web sudah punya disclaimer 'not a medical device'"
   web hanya punya "This result is an aid, not a diagnosis"
   -> 1 dari 4, mobile punya keempatnya

7  ambang kontras scrollbar saya tetapkan 4,5:1
   WCAG 1.4.11 = 3:1 untuk elemen non-teks

8  "analyze 213 ms per citra" untuk mobile
   sebenarnya 248-263 ms, resize 32 ms bukan 7 ms

9  push langsung ke master mobile, tanpa PR
   tidak ada proteksi di repo privat
```

Pola yang berlaku: temuan yang dibaca langsung dari berkas hampir selalu
benar. Temuan yang diingat dari sesi sebelumnya sering salah.

### Aturan yang sudah disepakati

```
Token tidak pernah masuk berkas, chat, atau commit
gh CLI di C:\Program Files\GitHub CLI\gh.exe, token di keyring Windows
build_index DILARANG, data/index.csv 669 baris sudah benar dan ter-commit
area_resize.dart di mobile tidak pernah ditulis ulang
confidence int8 tidak boleh ditampilkan sebagai persen
F1 1,0000 adalah syarat mutlak, tidak ditawar untuk demi ukuran
Agent tidak pernah merge, owner merge sendiri
```

### Peringatan teknis

```
.git push ke main di web DITOLAK, harus lewat PR
gh pr merge tanpa --admin, jangan pakai --admin kecuali diminta
Vim terbuka saat git pull butuh merge commit
  sudah di-set: git config --global core.editor "notepad"
  dan pakai --no-edit
.git_prompts/ di .gitignore, file prompt tidak masuk git
.agent/ di .gitignore, ~180 file, di-github tidak akan terlihat
```

---

## 9. VERIFIKASI YANG BISA DIJALANKAN ULANG

```powershell
# mobile, fokus utama
cd D:\2026\Workspace\PersonalApps\.MobileAppProjects\BacteriaCV
git checkout main
git pull --no-edit
git log --oneline -3
git status --porcelain
flutter analyze --fatal-infos
flutter test
# harus: 56 passed

# model, kalau perlu konfirmasi
Get-FileHash ..\..\BacteriaCV\checkpoints\heads.pt -Algorithm SHA256
# harus diawali 46F194B368603C3A
```

Perhatikan: jangan ada proses pytest lain yang berjalan bersamaan. Itu
menyebabkan kegagalan sporadis sebelum Task Z2.

Web sudah selesai, tapi kalau perlu dicek:

```powershell
cd D:\2026\Workspace\PersonalApps\BacteriaCV
& .\.venv\Scripts\python.exe -m pytest tests -q
# harus: 381 passed
```

---

## 10. LANGKAH LANGSUNG BERIKUTNYA

```
1  baca AGENTS.md dan docs/CODEX.md di repo BacteriaCV
2  cek git status di BacteriaCVMobile
3  selesaikan pekerjaan yang belum commit di branch
   layout-small-screens, atau laporkan dan biarkan
4  buat branch baru, commit, push, pull request
5  owner merge
6  setelah layout selesai: build AAB, lalu Play Store
```

Tidak ada yang mendesak. Owner menganggap web selesai, jadi seluruh
energi ke mobile.