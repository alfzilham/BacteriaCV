## Ringkasan

Bangun ulang BacteriaCV dari desain final sebagai monorepo MVC: situs (Next.js), API (FastAPI), aplikasi Android (Flutter).

## Lingkup

- `apps/site`: landing, dokumentasi, changelog dari GitHub Releases, WebApp `/app`, PWA, EN/ID, tema Light/Dark/System.
- `apps/api`: kode v1 dipindahkan tanpa mengubah kontrak, ditambah CORS (`CORS_ORIGINS`), `railway.json`, `.env.example`.
- `apps/mobile`: struktur `models/views/controllers/theme`, token desain, ThemeMode.system.
- CI: Python 3.11, 3.12, 3.13, build situs, analyze Flutter, kebijakan em-dash.
