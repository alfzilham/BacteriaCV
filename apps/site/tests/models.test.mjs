import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

// Tes memakai nilai literal dari desain, bukan nilai yang diturunkan dari modul.
const src = (p) => readFileSync(new URL(`../src/models/${p}`, import.meta.url), "utf8");

test("ambang keyakinan sesuai desain", () => {
  const code = src("confidence.ts");
  assert.match(code, />= 0\.8/);
  assert.match(code, />= 0\.6/);
});

test("teks tetap disclaimer dan batasan ada persis", () => {
  const copy = src("copy.ts");
  assert.ok(copy.includes("This result is an aid, not a diagnosis."));
  assert.ok(copy.includes("Segmentation groups cells, it does not isolate individual cells."));
});

test("tidak ada em-dash di kode sumber", () => {
  for (const f of ["copy.ts", "confidence.ts", "prediction.ts", "changelog.ts"]) assert.ok(!src(f).includes("\u2014"), f);
});

test("pemetaan tag rilis", () => {
  const code = src("changelog.ts");
  assert.ok(code.includes('"web-v"') && code.includes('"mobile-v"'));
});
