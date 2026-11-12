# apps/site

Situs BacteriaCV: landing, dokumentasi, changelog, dan WebApp (`/app`).
Next.js App Router, TypeScript, tanpa pustaka UI tambahan.

Struktur MVC:

- Model: `src/models` (kamus bahasa, token, data changelog, kontrak API).
- View: `src/views` (komponen tampilan) dan `src/app` (rute).
- Controller: `src/controllers` (hook tema, bahasa, unggah) dan `src/app/api` (webhook).

Jalankan: `npm install`, lalu `npm run dev`. Tes: `npm test`. Build: `npm run build`.
Variabel lingkungan: lihat `.env.example`.
