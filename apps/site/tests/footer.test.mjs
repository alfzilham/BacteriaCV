import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const text = (p) => readFileSync(new URL(`../${p}`, import.meta.url), "utf8");

// Kontras WCAG ditulis ulang di sini, tidak diimpor dari kode situs.
const lum = (hex) => {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const f = (v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
  return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
};
const ratio = (a, b) => {
  const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};
const FOOT_BG = "#08090A";

test("penghitung kontras mendeteksi warna lama footer yang gagal di tema gelap", () => {
  assert.ok(ratio("#6B7078", FOOT_BG) < 4.5);
  assert.ok(Math.abs(ratio("#FFFFFF", "#000000") - 21) < 0.01);
});

test("teks kecil footer memakai warna tetap dengan kontras minimal 4,5", () => {
  const css = text("src/app/globals.css");
  assert.ok(css.includes(".foot .col{font-size:13px;color:#8A9098"));
  const tsx = text("src/views/Chrome.tsx");
  assert.ok(tsx.includes('fontSize: 13, color: "#8A9098"'));
  assert.ok(!tsx.includes("var(--edge)"), "footer tidak boleh mengikuti variabel tema");
  assert.ok(ratio("#8A9098", FOOT_BG) >= 4.5);
});

test("fokus keyboard di footer terlihat pada tema terang dan gelap", () => {
  const css = text("src/app/globals.css");
  assert.ok(css.includes(".foot :focus-visible{outline-color:#F5F6F7}"));
  assert.ok(ratio("#F5F6F7", FOOT_BG) >= 3);
});
