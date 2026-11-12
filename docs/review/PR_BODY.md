## Ringkasan

Monorepo MVC untuk BacteriaCV (situs, API, Android). Lihat Issue terkait.

## Perubahan

- Situs Next.js baru di `apps/site`.
- API v1 dipindah ke `apps/api`, kontrak tidak berubah, SHA-256 `heads.pt` sama (46f194b3...bbd76).
- Flutter dirapikan ke MVC dan memakai token desain.
- CI dan skrip kebijakan di `.github/workflows` dan `tools/`.

## Validasi

Dijalankan di sesi penulisan: `py_compile apps/api/app/main.py`, `node --test` untuk tes model situs (4 lulus), pemeriksaan em-dash. BELUM dijalankan (tidak ada jaringan dan toolchain di sesi itu): `npm run build`, `npm run lint`, `pytest`, `flutter analyze`, `flutter test`. Jalankan atau tunggu CI sebelum menandai siap.

## Deployment Checklist

- Vercel root `apps/site`; env `NEXT_PUBLIC_API_URL`, `GITHUB_WEBHOOK_SECRET`.
- Railway root `apps/api`; env `CORS_ORIGINS`.
- Webhook GitHub Release ke `/api/webhooks/github`.
- Ekspor model ONNX dari `heads.pt` untuk Android (tidak di-commit).
