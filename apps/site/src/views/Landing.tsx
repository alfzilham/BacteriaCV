"use client";

import Link from "next/link";
import { usePrefs } from "@/controllers/prefs";
import { Footer, Header } from "./Chrome";
import { Splash } from "./Splash";

const SAMPLE = "/samples/contoh-1-bacilli-positif.png";
const Arrow = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--paper)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M7 17L17 7M8 7h9v9" /></svg>
);

export function Landing() {
  const { t } = usePrefs();
  const stages = ["st1", "st2", "st3", "st4", "st5"] as const;
  return (
    <div className="wrap">
      <Splash />
      <Header current="home" />
      <section style={{ display: "flex", flexWrap: "wrap", gap: 20, padding: "0 24px" }}>
        <div style={{ flex: "1 1 520px", minWidth: 0, display: "flex", flexDirection: "column", justifyContent: "space-between", gap: 28, padding: "24px 12px 8px" }}>
          <div>
            <h1 style={{ margin: 0, fontSize: "clamp(44px,7vw,96px)", lineHeight: 0.98, fontWeight: 400, letterSpacing: "-.05em", overflowWrap: "break-word" }}>{t("h1")}</h1>
            <p style={{ margin: "24px 0 0", maxWidth: 520, fontSize: 18, lineHeight: 1.6, color: "var(--t2)" }}>{t("lead")}</p>
            <div style={{ marginTop: 24, display: "flex", gap: 10, flexWrap: "wrap" }}>
              <Link className="pill k" href="/app" style={{ minHeight: 48 }}>{t("cta1")}</Link>
              <Link className="pill o" href="/docs" style={{ minHeight: 48 }}>{t("cta2")}</Link>
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {(["p1", "p2", "p3", "p4"] as const).map((k) => <span key={k} className="tag">{t(k)}</span>)}
          </div>
        </div>
        <div style={{ flex: "1.1 1 560px", minWidth: 0, position: "relative", borderRadius: 32, overflow: "hidden", background: "var(--tile)", aspectRatio: "4/3.2" }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={SAMPLE} alt="Gram stain microscope field with purple rod-shaped cells" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }} />
          <div style={{ position: "absolute", left: 14, right: 14, bottom: 14, borderRadius: 24, background: "var(--frost)", backdropFilter: "blur(14px)", padding: "16px 16px 16px 22px", display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ flex: 1, minWidth: 0 }}><div className="cap">{t("hp_cap")}</div><div style={{ fontSize: 28, letterSpacing: "-.03em", marginTop: 2 }}>{t("hp_res")}</div></div>
            <span className="round"><Arrow /></span>
          </div>
        </div>
      </section>

      <section id="features" className="sect">
        <h2 className="h2">{t("f_h")}</h2>
        <div className="cards">
          {([["f1_t", "f1_d", true], ["f2_t", "f2_d", false], ["f3_t", "f3_d", false]] as const).map(([a, b, dark]) => (
            <div key={a} style={{ borderRadius: 28, background: dark ? "var(--ink)" : "var(--tile)", color: dark ? "var(--paper)" : "var(--ink)", padding: 28, minHeight: 240, display: "flex", flexDirection: "column", justifyContent: "space-between", gap: 24 }}>
              <div style={{ fontSize: 36, letterSpacing: "-.04em", lineHeight: 1 }}>{t(a)}</div>
              <div style={{ fontSize: 15, lineHeight: 1.55, color: dark ? "var(--ink2)" : "var(--t2)" }}>{t(b)}</div>
            </div>
          ))}
        </div>
      </section>

      <section id="how" className="sect">
        <h2 className="h2" style={{ marginBottom: 12 }}>{t("h_h")}</h2>
        <p style={{ margin: "0 0 24px", fontSize: 16, lineHeight: 1.6, color: "var(--t2)", maxWidth: 560 }}>{t("h_d")}</p>
        <div className="steps">
          {stages.map((k, i) => (
            <div key={k} style={{ minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ borderRadius: 24, overflow: "hidden", background: "var(--tile)", aspectRatio: "4/3" }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={SAMPLE} alt="" style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} />
              </div>
              <div style={{ padding: "0 6px" }}><div className="cap">{i + 1} / 5</div><div style={{ fontSize: 18, letterSpacing: "-.02em", marginTop: 2 }}>{t(k)}</div></div>
            </div>
          ))}
        </div>
      </section>

      <section id="platforms" className="sect">
        <h2 className="h2">{t("pl_h")}</h2>
        <div className="cards">
          <div className="tile" style={{ display: "flex", flexDirection: "column", gap: 14, minHeight: 260 }}>
            <div style={{ fontSize: 36, letterSpacing: "-.04em", lineHeight: 1 }}>{t("pw_t")}</div>
            <div style={{ fontSize: 15, lineHeight: 1.55, color: "var(--t2)", maxWidth: 420 }}>{t("pw_d")}</div>
            <div style={{ marginTop: "auto" }}><Link className="pill o" href="/app">{t("cta1")}</Link></div>
          </div>
          <div className="tile" style={{ display: "flex", flexDirection: "column", gap: 14, minHeight: 260 }}>
            <div style={{ fontSize: 36, letterSpacing: "-.04em", lineHeight: 1 }}>{t("pa_t")}</div>
            <div style={{ fontSize: 15, lineHeight: 1.55, color: "var(--t2)", maxWidth: 420 }}>{t("pa_d")}</div>
            <div style={{ marginTop: "auto" }}>
              {/* Tombol demo: belum ada berkas APK atau Google Play. */}
              <button type="button" className="pill k" style={{ minHeight: 48 }}>{t("pa_btn")}</button>
              <div className="cap" style={{ marginTop: 10 }}>{t("pa_note")}</div>
            </div>
          </div>
        </div>
      </section>

      <section id="limits" className="sect">
        <h2 className="h2">{t("l_h")}</h2>
        <div className="cards">
          <div className="warn"><strong style={{ fontWeight: 600 }}>{t("disc")}</strong></div>
          <div className="warn">{t("lim_seg")}</div>
          <div className="warn">{t("l_test")}</div>
        </div>
      </section>

      <section id="open" className="sect">
        <div className="tile" style={{ display: "flex", flexWrap: "wrap", gap: 24, alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ flex: "1 1 300px", minWidth: 0 }}>
            <div style={{ fontSize: 40, letterSpacing: "-.04em", lineHeight: 1.05 }}>{t("o_h")}</div>
            <div style={{ marginTop: 12, fontSize: 15, lineHeight: 1.55, color: "var(--t2)", maxWidth: 560 }}>{t("o_d")}</div>
          </div>
          <a className="pill k" href="https://github.com/alfzilham/BacteriaCV">{t("o_gh")}</a>
        </div>
      </section>
      <Footer />
    </div>
  );
}
