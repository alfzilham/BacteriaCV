import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";

// Nilai literal dari keputusan desain, bukan nilai yang diimpor dari modul yang diuji.
const src = (p) => readFileSync(new URL(`../src/${p}`, import.meta.url), "utf8");

test("halaman /app tidak lagi memuat Footer, baris info dan disclaimer tetap ada", () => {
  const web = src("views/WebApp.tsx");
  assert.ok(!web.includes("<Footer"));
  assert.ok(!/import \{[^}]*Footer[^}]*\} from/.test(web));
  assert.ok(web.includes('t("disc")'));
  assert.ok(web.includes("20 MB"));
});

test("demo punya tepat empat contoh dengan id sesuai nama berkas", () => {
  const demo = src("models/demo.ts");
  for (const id of ["contoh-1-bacilli-positif", "contoh-2-bacilli-negatif", "contoh-3-cocci-positif", "contoh-4-cocci-negatif"]) {
    assert.ok(demo.includes(`id: "${id}"`), id);
    assert.ok(existsSync(new URL(`../public/samples/${id}.png`, import.meta.url)), id);
  }
  assert.equal((demo.match(/\bid: "/g) ?? []).length, 4);
  assert.ok(demo.includes('"/demo"'));
});

test("banner offline berdasarkan /api/health dan batas waktu 4 detik", () => {
  const h = src("controllers/health.ts");
  assert.ok(h.includes("/api/health"));
  assert.ok(h.includes("TIMEOUT_MS = 4000"));
  assert.ok(src("views/WebApp.tsx").includes('health === "offline"'));
});

test("riwayat hanya menyimpan label, maksimal lima, dan tidak menyimpan gambar", () => {
  const h = src("controllers/history.ts");
  assert.ok(h.includes('HISTORY_KEY = "bcv-history"'));
  assert.ok(h.includes("HISTORY_MAX = 5"));
  assert.ok(h.includes("sessionStorage"));
  for (const bad of ["panels", "File", "base64", "localStorage"]) assert.ok(!h.includes(bad), bad);
});

test("legenda keyakinan memakai ambang desain", () => {
  const copy = src("models/copy.ts");
  assert.ok(copy.includes("High: 0.80 or above"));
  assert.ok(copy.includes("Medium: 0.60 to 0.79"));
  assert.ok(copy.includes("Low: below 0.60"));
});

test("kontrak /api/predict tidak berubah: kunci respons tetap", () => {
  const p = src("models/prediction.ts");
  for (const k of ["shape_label", "gram_label", "segmentation_ok", "stage_index", "stage_names", "panels", "notes_text", "disclaimer", "failed_stages"]) {
    assert.ok(p.includes(k), k);
  }
});

test("skrip ekspor ada dan menulis ke public/demo", () => {
  const s = readFileSync(new URL("../../../tools/export-stages.ps1", import.meta.url), "utf8");
  assert.ok(s.includes("apps\\site\\public\\demo"));
  assert.ok(s.includes("/api/predict?stage=0"));
});

test("tidak ada em-dash di berkas baru atau yang diubah", () => {
  for (const f of ["models/demo.ts", "controllers/health.ts", "controllers/history.ts", "controllers/analyze.ts", "views/SampleGallery.tsx", "views/ApiBanner.tsx", "views/InfoPanels.tsx", "views/WebApp.tsx", "models/copy.ts"]) {
    assert.ok(!src(f).includes("\u2014"), f);
  }
});
