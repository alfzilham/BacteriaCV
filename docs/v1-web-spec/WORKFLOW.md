# WORKFLOW: Alur Kerja Owner dan Claude Code

Dokumen ini mengatur cara kerja, bukan isi produk. Keputusan produk dan teknis
tetap mengikuti `SPEC.md`, `ARCHITECTURE.md`, `DESIGN.md`, `PRD.md`, dan
`CONTEXT.md`.

## 1. Peran

| Pihak | Peran |
|-------|-------|
| Owner (Alfiz Ilham) | Memberi tugas, menyetujui rencana, menjalankan perintah git/gh, menggabungkan PR |
| Claude (chat) | Membaca codebase, menyusun Context Summary untuk Claude Code |
| Claude Code | Membuat rencana, mengimplementasikan setelah disetujui, menyiapkan perintah git/gh |

## 2. Urutan kerja satu tugas

1. Owner memberi tugas.
2. Claude Code memeriksa `git status` dan membaca dokumen sumber yang relevan.
3. Claude Code menyampaikan pemahaman tugas secara poin per poin, lalu rencana:
   lingkup, berkas yang diperkirakan berubah, validasi, dan risiko.
4. Claude Code menunggu persetujuan eksplisit owner. Tidak ada kode yang diubah
   sebelum owner menyetujui.
5. Claude Code mengimplementasikan hanya lingkup yang disetujui.
6. Claude Code menjalankan validasi: `python -m pytest tests -q` dan, bila
   menyentuh lapisan web, menjalankan server lalu memeriksa `/api/health`.
   Validasi yang tidak bisa dijalankan dilaporkan sebagai "belum dicek".
7. Claude Code melaporkan berkas yang berubah, hasil validasi, dan risiko
   yang tersisa.
8. Claude Code memberikan perintah git/gh siap salin (bagian 3).
9. Owner menjalankan perintah, lalu menggabungkan PR di GitHub.

## 3. Perintah git/gh setelah setiap revisi kode

Diberikan dalam satu jalur saja, tanpa pilihan A/B. Asumsinya PR sebelumnya
sudah digabung, sehingga setiap perubahan dimulai dari `main` terbaru pada
branch baru.

1. Satu blok PowerShell berisi: buat branch baru dari `main` terbaru, commit
   satu berkas per commit, push, buat Issue (`gh issue create`), lalu buat PR
   (`gh pr create`) yang menautkan Issue dengan `Closes #N`.
2. Dua kalimat review, masing-masing di kotak kode sendiri: satu untuk review
   Comment, satu untuk review Approval.
3. Perintah `gh release create` hanya setelah lima PR tergabung sejak release
   terakhir. Tag naik di patch sampai 9 (v1.0.0 sampai v1.0.9), lalu minor naik
   dan patch kembali ke 0 (v1.1.0).

Aturan PowerShell:

- Body Issue dan PR yang multiline ditulis ke berkas sementara lalu dikirim
  dengan `--body-file`, bukan `--body $var`.
- Skrip `.ps1` disimpan sebagai UTF-8 dengan BOM, string dan komentar hanya
  ASCII, dan tidak ada backtick di dalam here-string.

## 4. Commit

- Satu commit hanya berisi satu berkas.
- Format pesan: Conventional Commit, misalnya `feat:`, `fix:`, `refactor:`,
  `docs:`, `test:`, `chore:`.
- Jangan commit `.env`, kredensial, log, `.venv`, `data/raw`, `data/features`,
  atau artefak build.
- Jangan memakai karakter em-dash di berkas mana pun.

## 5. Batasan

- Agent tidak pernah menggabungkan PR, tidak force push, dan tidak push
  langsung ke `main`.
- Kontrak API (`/api/health`, `/api/predict`, `/api/report`, beserta field
  JSON-nya) dan format `checkpoints/heads.pt` tidak berubah tanpa persetujuan
  owner.
- Lookup table hanya dipakai saat training. Bobot kelas hanya dihitung dari
  data train.
- Tidak ada tes yang dihapus atau dilemahkan agar tugas lulus.

## 6. Struktur yang perlu diketahui

- Model: paket `bacteriacv/` (preprocess, model, infer, visualize, train,
  evaluate, datasets).
- View: `app/views/templates/index.html` dan `app/views/static/`.
- Controller: `app/controllers/`, dibantu `app/services/`.
- Titik masuk server: `app.main:app`. Di PowerShell jalankan
  `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`.
