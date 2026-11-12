# BacteriaCV

Bentuk sel dan status Gram bakteri dari satu citra mikroskop. Sumber terbuka, dilatih dengan data publik DIBaS.

Hasil model adalah alat bantu, bukan diagnosis. Segmentasi mengelompokkan sel, bukan memisahkan sel satu per satu.

## Struktur (monorepo, MVC di tiap aplikasi)

| Folder | Isi | Deploy |
|--------|-----|--------|
| `apps/site` | Landing, dokumentasi, changelog, WebApp `/app` (Next.js) | Vercel |
| `apps/api` | Inferensi (FastAPI + PyTorch). Model: `bacteriacv/`, controller: `app/controllers`, service: `app/services` | Railway |
| `apps/mobile` | Aplikasi Android (Flutter, ONNX di perangkat). `lib/models`, `lib/views`, `lib/controllers`, `lib/theme` | APK |
| `docs/` | Dokumen sumber dan arsip spesifikasi v1 | - |
| `.github/workflows` | CI | - |

## Menjalankan

```
# API
cd apps/api && python -m pip install -r requirements.txt && uvicorn app.main:app --port 8000
# Situs
cd apps/site && npm install && npm run dev
# Android
cd apps/mobile && flutter pub get && flutter test
```

Variabel lingkungan: `apps/api/.env.example` dan `apps/site/.env.example`.
Model ONNX untuk Android tidak di-commit; ekspor dari `apps/api/checkpoints/heads.pt` lalu taruh di `apps/mobile/assets/spike/model_int8.onnx`.

## Deploy

- Vercel: root directory `apps/site`. Set `NEXT_PUBLIC_API_URL` ke URL Railway, `GITHUB_WEBHOOK_SECRET`.
- Railway: root directory `apps/api`. Set `CORS_ORIGINS` ke domain Vercel. Health check `/api/health`.
- Webhook GitHub (event Release) ke `https://<domain-vercel>/api/webhooks/github`. Tag `web-v*` untuk web, `mobile-v*` untuk Android.

Lisensi: MIT.
