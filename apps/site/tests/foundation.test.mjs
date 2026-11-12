import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

// Nilai literal dari keputusan desain, bukan nilai yang diimpor dari modul yang diuji.
const src = (p) => readFileSync(new URL(`../src/${p}`, import.meta.url), "utf8");

test("kunci tema tetap bcv-theme dan dipakai ThemeProvider", () => {
  assert.ok(src("controllers/prefs.tsx").includes('export const THEME_KEY = "bcv-theme";'));
  const tp = src("controllers/ThemeProvider.tsx");
  assert.ok(tp.includes("storageKey={THEME_KEY}"));
  assert.ok(tp.includes('defaultTheme="system"') && tp.includes("enableSystem"));
});

test("atribut tema mencakup data-theme untuk tokens.css dan class untuk efek wipe", () => {
  const tp = src("controllers/ThemeProvider.tsx");
  assert.ok(tp.includes('"class"') && tp.includes('"data-theme"'));
  assert.ok(src("models/tokens.css").includes('html[data-theme="dark"]'));
});

test("skrip boot tema lama sudah dihapus dari layout dan prefs", () => {
  assert.ok(!src("app/layout.tsx").includes("THEME_BOOT"));
  assert.ok(!src("controllers/prefs.tsx").includes("THEME_BOOT"));
});

test("layout memasang ThemeProvider dan SmoothScroll untuk semua halaman", () => {
  const layout = src("app/layout.tsx");
  assert.ok(layout.includes("<ThemeProvider>") && layout.includes("<SmoothScroll />"));
});

test("Lenis dan splash menghormati prefers-reduced-motion", () => {
  assert.ok(src("views/SmoothScroll.tsx").includes("(prefers-reduced-motion: reduce)"));
  assert.ok(src("views/Splash.tsx").includes("useReducedMotion"));
});

test("splash sekali per sesi lewat sessionStorage dan hanya di landing", () => {
  assert.ok(src("views/Splash.tsx").includes('export const SPLASH_KEY = "bcv-splash";'));
  assert.ok(src("views/Splash.tsx").includes("sessionStorage"));
  assert.ok(src("views/Landing.tsx").includes("<Splash />"));
  for (const f of ["views/DocsShell.tsx", "views/WebApp.tsx", "views/Chrome.tsx"]) assert.ok(!src(f).includes("<Splash"), f);
});

test("toggle tema mempertahankan tiga mode: terang, gelap, sistem", () => {
  const c = src("views/ThemeToggle.tsx");
  assert.ok(c.includes('toggleTheme(effective === "dark" ? "light" : "dark")'));
  assert.ok(c.includes('setTheme("system")'));
});

test("tidak ada em-dash di berkas baru", () => {
  for (const f of ["controllers/ThemeProvider.tsx", "controllers/prefs.tsx", "views/SmoothScroll.tsx", "views/Splash.tsx", "views/ThemeToggle.tsx", "views/Chrome.tsx", "app/layout.tsx"]) {
    assert.ok(!src(f).includes("\u2014"), f);
  }
});
