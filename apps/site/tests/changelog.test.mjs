import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

// Nilai literal, bukan nilai yang diimpor dari modul yang diuji.
const src = (p) => readFileSync(new URL(`../src/${p}`, import.meta.url), "utf8");

test("changelog tidak memuat teks placeholder", () => {
  for (const f of ["models/changelog.ts", "controllers/changelog.ts", "views/ChangelogView.tsx"]) {
    const code = src(f);
    assert.ok(!code.includes("PLACEHOLDERS"), `${f} masih memuat PLACEHOLDERS`);
    assert.ok(!code.includes("[release title]"), `${f} masih memuat [release title]`);
    assert.ok(!code.includes('"[date]"'), `${f} masih memuat [date]`);
  }
});

test("controller mengembalikan daftar kosong saat fetch gagal atau tidak ada tag cocok", () => {
  const code = src("controllers/changelog.ts");
  assert.equal((code.match(/return \[\];/g) ?? []).length, 2);
  assert.ok(code.includes("return toReleases(await res.json());"));
});

test("keadaan kosong punya teks dua bahasa dan tautan ke GitHub Releases", () => {
  const copy = src("models/copy.ts");
  const line = (k) => copy.split("\n").find((l) => l.trim().startsWith(`${k}:`)) ?? "";
  for (const k of ["cl_empty", "cl_all"]) {
    const l = line(k);
    assert.equal((l.match(/"[^"]+"/g) ?? []).length, 2, `${k} harus berisi dua teks`);
    assert.ok(!l.includes(String.fromCharCode(0x2014)), `${k} tidak boleh memuat em-dash`);
  }
  const view = src("views/ChangelogView.tsx");
  assert.ok(view.includes('href="https://github.com/alfzilham/BacteriaCV/releases"'));
  assert.ok(view.includes('rel="noreferrer"'));
});

test("tag v<angka> dikenali sebagai rilis gabungan, web-v dan mobile-v tetap", () => {
  const m = src("models/changelog.ts");
  assert.ok(m.includes('"web" | "android" | "all"'));
  assert.ok(m.includes('tag.startsWith("web-v") ? "web"'));
  assert.ok(m.includes('tag.startsWith("mobile-v") ? "android"'));
  assert.ok(m.includes(String.raw`/^v\d/.test(tag) ? "all" : null`));
});

test("rilis gabungan tampil di setiap filter dan berlabel All", () => {
  const view = src("views/ChangelogView.tsx");
  assert.ok(view.includes('r.platform === f || r.platform === "all"'));
  assert.ok(view.includes('r.platform === "android" ? t("f_and") : t("f_all")'));
});
