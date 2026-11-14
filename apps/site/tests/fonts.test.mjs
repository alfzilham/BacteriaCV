import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";

const url = (p) => new URL(`../${p}`, import.meta.url);
const text = (p) => readFileSync(url(p), "utf8");
const FONT = "src/app/fonts/hanken-grotesk-latin-wght-normal.woff2";

test("situs tidak memuat font atau berkas apa pun dari host Google", () => {
  for (const p of ["src/app/layout.tsx", "src/app/globals.css"]) {
    const s = text(p);
    assert.ok(!s.includes("googleapis"), `${p} masih menyebut googleapis`);
    assert.ok(!s.includes("gstatic"), `${p} masih menyebut gstatic`);
  }
});

test("font Hanken Grotesk dilayani dari repo lewat next/font/local", () => {
  const layout = text("src/app/layout.tsx");
  assert.ok(layout.includes('from "next/font/local"'));
  assert.ok(layout.includes("./fonts/hanken-grotesk-latin-wght-normal.woff2"));
  assert.ok(layout.includes('variable: "--font-hanken"'));
  assert.ok(text("src/app/globals.css").includes("var(--font-hanken)"));
});

test("berkas font ada, berformat WOFF2, dan disertai catatan lisensi", () => {
  assert.ok(existsSync(url(FONT)), "berkas woff2 belum ada");
  const head = readFileSync(url(FONT)).subarray(0, 4).toString("latin1");
  assert.equal(head, "wOF2");
  assert.ok(text("src/app/fonts/NOTICE.md").includes("SIL Open Font License"));
});
