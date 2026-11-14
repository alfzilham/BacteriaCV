import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";

// Nilai literal dari kode API dan README, bukan nilai yang diimpor dari modul yang diuji.
const src = (p) => readFileSync(new URL(`../src/${p}`, import.meta.url), "utf8");
const has = (p) => existsSync(new URL(`../src/${p}`, import.meta.url));

const PAGES = [
  ["quick-start", "QuickStart"], ["how-it-works", "HowItWorks"], ["self-host", "SelfHost"],
  ["api", "ApiDocs"], ["limits", "Limits"], ["faq", "DocsFaq"], ["changelog", "ChangelogView"],
];

test("setiap halaman dokumentasi punya rute dan bilah sisi", () => {
  const shell = src("views/DocsShell.tsx");
  for (const [slug] of PAGES) {
    assert.ok(has(`app/docs/${slug}/page.tsx`), `rute ${slug}`);
    assert.ok(shell.includes(`"/docs/${slug}"`), `bilah sisi ${slug}`);
  }
});

test("setiap kunci teks yang dipakai halaman baru ada di kamus bahasa", () => {
  const copy = src("models/copy.ts");
  const views = ["HowItWorks", "SelfHost", "Limits", "DocsFaq", "ApiDocs", "QuickStart"].map((v) => src(`views/${v}.tsx`)).join("\n");
  const used = new Set([...views.matchAll(/\bt\("([a-z0-9_]+)"\)/g)].map((m) => m[1]));
  for (const k of ["st1", "sd1", "st5", "sd5", "hw_m1", "sh_e6", "lm_and", "faq_h"]) used.add(k);
  for (const k of used) assert.ok(new RegExp(`\\b${k}: \\[`).test(copy), `kunci ${k}`);
});

test("panduan jalankan sendiri memuat perintah dan variabel yang benar", () => {
  const v = src("views/SelfHost.tsx");
  for (const s of ["python -m venv .venv", "python -m pip install -r requirements.txt", "uvicorn app.main:app --port 8000", "npm install", "npm run dev", "CORS_ORIGINS", "PORT", "RATE_LIMIT_PER_MINUTE", "RATE_LIMIT_TRUST_FORWARDED", "NEXT_PUBLIC_API_URL", "GITHUB_REPO", "GITHUB_WEBHOOK_SECRET", "GITHUB_TOKEN"]) {
    assert.ok(v.includes(s), s);
  }
  const copy = src("models/copy.ts");
  for (const s of ["apps/site", "apps/api", "/api/webhooks/github", "tools/export-stages.ps1", "apps/site/public/demo", "502"]) assert.ok(copy.includes(s), s);
});

test("kunci lingkungan di panduan sama dengan .env.example situs", () => {
  const env = readFileSync(new URL("../.env.example", import.meta.url), "utf8");
  for (const k of ["NEXT_PUBLIC_API_URL", "GITHUB_WEBHOOK_SECRET", "GITHUB_TOKEN", "GITHUB_REPO"]) assert.ok(env.includes(k), k);
});

test("halaman API mendaftar semua kode galat dan tiga contoh pemanggilan", () => {
  const copy = src("models/copy.ts");
  const line = copy.split("\n").find((l) => l.includes("d_err_d:"));
  assert.ok(line, "baris d_err_d");
  // Dua bahasa pada satu baris, jadi tiap kode harus muncul tepat dua kali.
  for (const code of ["400", "413", "415", "429", "500", "503"]) assert.equal(line.split(`${code} `).length - 1, 2, code);
  const a = src("views/ApiDocs.tsx");
  for (const s of ["curl -X POST", "requests.post", "FormData", "/api/predict?stage=3", "failed_stages", "notes_text", "panels"]) assert.ok(a.includes(s), s);
});

test("angka data dan model di halaman cara kerja cocok dengan README dan index.csv", () => {
  const copy = src("models/copy.ts");
  for (const s of ["669 images from 32 bacterial species", "467 for training", "136 for validation", "66 for testing", "ResNet-50", "2048 inputs", "heads.pt"]) assert.ok(copy.includes(s), s);
});

test("tautan banner API dan FAQ mengarah ke panduan jalankan sendiri", () => {
  assert.ok(src("views/ApiBanner.tsx").includes('href="/docs/self-host"'));
  assert.ok(src("models/copy.ts").includes("The Run it yourself guide explains how."));
});

test("berkas dokumentasi tidak memuat em-dash", () => {
  const dash = String.fromCharCode(0x2014);
  const files = ["models/copy.ts", "views/DocBits.tsx", "views/HowItWorks.tsx", "views/SelfHost.tsx", "views/Limits.tsx", "views/DocsFaq.tsx", "views/ApiDocs.tsx", "views/QuickStart.tsx", "views/DocsShell.tsx"];
  for (const f of files) assert.ok(!src(f).includes(dash), f);
});
