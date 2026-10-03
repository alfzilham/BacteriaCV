# AGENT.md: Instruksi untuk Agen di Repositori

## 1. Peran
| Agen | Alat | Peran | Kewenangan |
|------|------|-------|------------|
| opencode | opencode | Penulis kode | Menulis dan mengubah kode sesuai SPEC, ARCHITECTURE, DESIGN |
| Codex CLI | OpenAI Codex CLI | Auditor | Mengaudit; memperbaiki berkas tes yang disetujui pemilik; menyusun laporan final |

Auditor dan penulis tidak boleh sama untuk satu tugas yang sama.

## 2. Sumber Kebenaran
Urutan prioritas bila terjadi konflik:
1. SPEC.md
2. ARCHITECTURE.md
3. DESIGN.md
4. PRD.md dan CONTEXT.md (berstatus draf hingga divalidasi)
5. AGENT.md
6. Kode yang ada

Agen tidak boleh mengubah keputusan yang tercatat di SPEC.md tanpa persetujuan eksplisit dari pemilik proyek.

## 3. Aturan untuk opencode (Penulis)
1. Kerjakan satu tugas per perubahan. Jangan menggabungkan refactor dengan fitur baru.
2. Ikuti antarmuka pada DESIGN.md. Bila perlu mengubah antarmuka, tulis usulan perubahan dan tunggu persetujuan.
3. Jangan mengubah lookup table [L] atau aturan class weighting tanpa persetujuan.
4. Jangan memakai data uji untuk keputusan apa pun selama pelatihan.
5. Jangan menyimpan kredensial, token, atau path absolut di kode.
6. Setiap tugas selesai harus disertai tes sesuai DESIGN.md Bagian 5.
7. Setelah selesai, serahkan ringkasan perubahan, daftar berkas, dan prompt audit kepada pemilik proyek.
8. Prompt audit memuat konteks, acuan, daftar berkas, dan fokus pemeriksaan. Prompt harus cukup
   lengkap agar auditor tidak perlu menebak ruang lingkup tugas.
9. Jangan menyatakan tugas selesai sebelum laporan final auditor menyatakan LULUS.

## 4. Aturan untuk Codex CLI (Auditor)
1. Audit wajib dilakukan sebelum tugas dianggap selesai.
2. Periksa kesesuaian dengan SPEC.md dan ARCHITECTURE.md.
3. Periksa kualitas dengan rubrik penilaian laporan: pemahaman konsep, inovasi dan relevansi, kedalaman teknis, dan sistematika.
4. Periksa secara khusus:
   - kebocoran data antara latih, validasi, dan uji;
   - penggunaan bobot kelas yang dihitung dari data selain data latih;
   - lookup table diterapkan hanya pada tahap pelatihan;
   - F1-score makro dihitung dan dilaporkan.
5. Format temuan: `[severity] berkas:baris - deskripsi - usulan perbaikan`.
   Severity: `kritis`, `mayor`, `minor`.
6. Auditor tidak mengubah kode produksi. Kode produksi hanya diubah oleh penulis, karena bila
   auditor mengubah berkas yang baru diaudit, kesalahannya bisa tertutupi sendiri dan hasil
   audit tidak lagi independen.
7. Auditor boleh memperbaiki berkas tes setelah pemilik proyek menyetujui usulannya secara
   eksplisit. Berkas tes bukan objek yang dinilai kelayakannya oleh audit.
8. Setelah audit selesai, auditor langsung menyusun daftar temuan dan meminta verifikasi kepada
   pemilik proyek. Auditor tidak menunggu diminta ulang.
9. Setiap permintaan verifikasi memuat per temuan: severity, berkas:baris, dampak atau risiko,
   dan usulan perbaikan konkret. Pemilik harus dapat memutuskan tanpa membaca kode.
10. Temuan kritis memblokir penyelesaian tugas hingga diperbaiki.
11. Verdict hanya dua: LULUS atau BELUM LULUS. LULUS diberikan bila tidak ada temuan kritis
    maupun mayor. Minor tidak memblokir, tetapi tetap wajib dicatat.
12. Setelah semua temuan yang disetujui diperbaiki, auditor menyusun satu laporan final yang
    menyebutkan status tiap temuan dan verdict akhir. Laporan inilah yang diteruskan pemilik
    kepada penulis.

## 5. Alur Kerja
```
Tugas dari pemilik
      |
      v
opencode: implementasi + tes
      |
      v
opencode: serah terima ringkasan, daftar berkas, dan prompt audit
      |
      v
Codex CLI: audit, hasilkan temuan
      |
      v
Codex CLI: minta verifikasi temuan kepada pemilik
      |
      v
Pemilik: setujui atau tolak tiap temuan
      |
      v
Codex CLI: terapkan perbaikan tes yang disetujui
      |
      v
Codex CLI: audit ulang bila ada temuan kritis atau mayor
      |
      v
Codex CLI: laporan final beserta verdict
      |
      v
Pemilik: teruskan laporan final kepada opencode
      |
      v
Tugas selesai bila verdict LULUS
```

Bila verdict BELUM LULUS, alur kembali ke tahap audit. Tidak ada batas jumlah putaran, tetapi
tiap putaran harus menghasilkan perubahan nyata atau penjelasan alasan temuan itu ditutup.

## 6. Batasan Umum
- Tidak ada agen yang mengunggah data DIBaS ke layanan eksternal.
- Tidak ada agen yang menjalankan pelatihan penuh tanpa konfirmasi, karena beban GPU konsumen.
- Setiap perubahan dicatat dalam riwayat git dengan pesan commit yang merujuk tugas.
- Data mentah DIBaS tidak di-commit. Yang di-commit hanya `data/index.csv` dan
  `data/raw/zips_manifest.csv` sebagai bukti provenance.

## 7. Format Pesan Commit
`[tipe] ringkasan singkat` dengan tipe: feat, fix, test, docs, refactor.

## 8. Format Laporan Final Auditor
Laporan final memuat lima bagian dengan urutan tetap:
1. **Ringkasan hasil** — apa yang sudah diverifikasi dan bagaimana cara memverifikasinya.
2. **Temuan** — temuan tersisa dengan format `[severity] berkas:baris - deskripsi - usulan`.
   Bila tidak ada, tulis "Tidak ada temuan".
3. **Status tiap temuan sebelumnya** — untuk setiap temuan dari putaran sebelumnya, sebutkan
   statusnya: ditutup, ditolak dengan alasan, atau masih terbuka.
4. **Kesuaian spesifikasi** — penilaian terhadap SPEC.md dan aturan di AGENT.md ini.
5. **Verdict** — LULUS atau BELUM LULUS, disertai alasan ringkas.
