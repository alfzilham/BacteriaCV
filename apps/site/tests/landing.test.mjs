import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

// Nilai literal dari keputusan owner, bukan nilai yang diimpor dari modul yang diuji.
const src = (p) => readFileSync(new URL(`../src/${p}`, import.meta.url), "utf8");

test("kontak memuat email, WhatsApp, LinkedIn, GitHub, Issues, dan Discussions persis", () => {
  const c = src("views/Contact.tsx");
  assert.ok(c.includes("mailto:alfizilham@gmail.com"));
  assert.ok(c.includes("https://wa.me/6287822065257"));
  assert.ok(c.includes("+62 878 2206 5257"));
  assert.ok(c.includes("https://www.linkedin.com/in/alfiz-ilham093a/"));
  assert.ok(c.includes("https://github.com/alfzilham/BacteriaCV/issues"));
  assert.ok(c.includes("https://github.com/alfzilham/BacteriaCV/discussions"));
  assert.ok(c.includes('href: "https://github.com/alfzilham"'));
});

test("FAQ punya enam pertanyaan dan jawaban", () => {
  const f = src("views/Faq.tsx");
  for (let i = 1; i <= 6; i++) assert.ok(f.includes(`"qa${i}_q", "qa${i}_a"`), `qa${i}`);
  assert.ok(f.includes("aria-expanded"));
  assert.ok(f.includes("AnimatePresence"));
});

test("landing memasang marquee, FAQ, kontak, dan reveal, dan splash tetap hanya di landing", () => {
  const l = src("views/Landing.tsx");
  for (const s of ["<Marquee", "<Faq />", "<Contact />", "<Reveal>", "<Splash />", 'id="faq"', 'id="contact"']) assert.ok(l.includes(s), s);
});

test("gambar tahap landing memakai hasil demo dengan fallback ke contoh", () => {
  const l = src("views/Landing.tsx");
  assert.ok(l.includes("demoPanelUrl(DEMO_SAMPLES[0].id, index)"));
  assert.ok(l.includes("onError"));
});

test("marquee, hover, dan akordeon menghormati prefers-reduced-motion", () => {
  const css = src("app/globals.css");
  assert.ok(css.includes(".mq-track{animation:none"));
  assert.ok(css.includes("(hover:hover) and (pointer:fine)"));
  assert.ok(src("views/Reveal.tsx").includes("useReducedMotion"));
  assert.ok(src("views/Faq.tsx").includes("useReducedMotion"));
});

test("teks mempertahankan disclaimer dan batasan, dan klaim akurasi memuat peringatan set uji kecil", () => {
  const copy = src("models/copy.ts");
  assert.ok(copy.includes("This result is an aid, not a diagnosis."));
  assert.ok(copy.includes("On the 66 held-out test images"));
  assert.ok(copy.includes("not as clinical validation"));
});

test("tidak ada em-dash di berkas baru atau yang diubah", () => {
  for (const f of ["views/Reveal.tsx", "views/Marquee.tsx", "views/Faq.tsx", "views/Contact.tsx", "views/Landing.tsx", "views/Chrome.tsx", "models/copy.ts", "app/globals.css"]) {
    assert.ok(!src(f).includes("\u2014"), f);
  }
});
