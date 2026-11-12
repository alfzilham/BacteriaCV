"use client";

import { useEffect, useRef, useState } from "react";
import { useAnalyze } from "@/controllers/analyze";
import { usePrefs } from "@/controllers/prefs";
import { dotsOf, levelOf } from "@/models/confidence";
import type { PredictResponse } from "@/models/prediction";
import { Controls, Footer } from "./Chrome";

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

function Dots({ level }: { level: "High" | "Medium" | "Low" }) {
  const n = dotsOf(level);
  return (
    <span role="img" aria-label={`Confidence level ${level}, ${n} of 3`} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, fontWeight: 500 }}>
      <span style={{ display: "flex", gap: 3 }}>{[1, 2, 3].map((i) => <i key={i} style={{ width: 7, height: 7, borderRadius: "50%", background: i <= n ? "var(--ink)" : "var(--dot)" }} />)}</span>
      {level}
    </span>
  );
}

function Result({ data, onReset }: { data: PredictResponse; onReset: () => void }) {
  const { t } = usePrefs();
  const [stage, setStage] = useState(0);
  const p = data.prediction;
  const sl = levelOf(p.shape_confidence);
  const gl = levelOf(p.gram_confidence);
  const low = sl === "Low" || gl === "Low";
  const stageNames = data.stage_names;
  return (
    <main style={{ display: "flex", flexWrap: "wrap", gap: 20, alignItems: "stretch", padding: "0 24px" }}>
      <section aria-label="Classification" style={{ flex: "1.15 1 520px", minWidth: 0, display: "flex", flexDirection: "column", gap: 24 }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 16 }}>
          <div className="rise" style={{ flex: "1 1 260px", borderRadius: 28, background: "var(--tile)", padding: 12, display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ borderRadius: 20, background: "var(--ink)", color: "var(--paper)", padding: "16px 20px" }}>
              <div style={{ fontSize: 12, color: "var(--ink2)" }}>{t("a_shape")}</div>
              <div style={{ fontSize: 36, letterSpacing: "-.03em", lineHeight: 1, marginTop: 6 }}>{cap(p.shape_label)}</div>
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", padding: "0 8px 4px" }}><Dots level={sl} /></div>
          </div>
          <div className="rise" style={{ flex: "1 1 260px", borderRadius: 28, border: "1px solid var(--line)", padding: 12, display: "flex", flexDirection: "column", gap: 12, animationDelay: "60ms" }}>
            <div style={{ borderRadius: 20, background: "var(--tile)", padding: "16px 20px" }}>
              <div style={{ fontSize: 12, color: "var(--t2)" }}>{t("a_gram")}</div>
              <div style={{ fontSize: 36, letterSpacing: "-.03em", lineHeight: 1, marginTop: 6, color: p.gram_label === "positive" ? "var(--violet)" : "var(--safr)" }}>{cap(p.gram_label)}</div>
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", padding: "0 8px 4px" }}><Dots level={gl} /></div>
          </div>
        </div>
        <div style={{ padding: "8px 8px 0" }}>
          <div style={{ fontSize: 14, color: "var(--t3)" }}>{t("a_summary")}</div>
          <h1 style={{ margin: "8px 0 0", fontSize: "clamp(44px,6vw,88px)", lineHeight: 0.98, fontWeight: 400, letterSpacing: "-.045em" }}>
            {cap(p.shape_label)}, Gram <span style={{ color: p.gram_label === "positive" ? "var(--violet)" : "var(--safr)" }}>{p.gram_label}</span>
          </h1>
          <p style={{ margin: "20px 0 0", maxWidth: 620, fontSize: 15, lineHeight: 1.6, color: "var(--t2)" }}>
            <strong style={{ color: "var(--ink)", fontWeight: 600 }}>{t("disc")}</strong> {low ? t("a_low") : ""}
          </p>
          <div style={{ marginTop: 16 }}><button type="button" className="pill k" onClick={onReset}>{t("a_another")}</button></div>
        </div>
        <div role="toolbar" aria-label="Visualisation stage" style={{ marginTop: "auto", display: "flex", flexWrap: "wrap", gap: 10 }}>
          {stageNames.map((n, i) => <button key={n} type="button" className="chip" aria-pressed={stage === i} onClick={() => setStage(i)}>{i + 1} {n}</button>)}
        </div>
      </section>
      <section aria-label="Segmentation visualisation" style={{ flex: "1 1 460px", minWidth: 0, position: "relative", borderRadius: 32, overflow: "hidden", background: "var(--tile)", minHeight: 560 }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={`data:image/png;base64,${data.panels[stage]}`} alt={stageNames[stage]} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }} />
        <div style={{ position: "absolute", left: 20, right: 20, bottom: 20, borderRadius: 28, background: "var(--frost)", backdropFilter: "blur(14px)", padding: "20px 24px" }}>
          <div className="cap">{t("a_stage")} {stage + 1} {t("a_of")}</div>
          <div style={{ fontSize: 28, letterSpacing: "-.03em", lineHeight: 1.1, marginTop: 4 }}>{stageNames[stage]}</div>
          <div style={{ fontSize: 13, lineHeight: 1.5, color: "var(--t2)", marginTop: 10, paddingTop: 10, borderTop: "1px solid var(--line)" }}>
            <strong style={{ color: "var(--ink)", fontWeight: 600 }}>{t("a_lim")}</strong> {t("lim_seg")}
          </div>
        </div>
      </section>
    </main>
  );
}

export function WebApp() {
  const { t } = usePrefs();
  const { state, run, reset } = useAnalyze();
  const [file, setFile] = useState<File | null>(null);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => { if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => undefined); }, []);
  const errText = state.status === "error" ? (state.reason === "network" ? t("a_net") : t("a_err")) + (state.detail ? ` (${state.detail})` : "") : "";
  return (
    <div className="wrap" style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <header className="hdr" style={{ paddingBottom: 20 }}>
        <a href="/" style={{ fontSize: 24, fontWeight: 600, letterSpacing: "-.03em", textDecoration: "none" }}>BacteriaCV</a>
        <div style={{ fontSize: 12, color: "var(--t3)", maxWidth: 190, lineHeight: 1.35 }}>{t("a_tag")}</div>
        <div style={{ marginLeft: "auto" }}><Controls /></div>
      </header>
      {state.status === "done" ? (
        <Result data={state.data} onReset={() => { reset(); setFile(null); }} />
      ) : (
        <main style={{ padding: "0 24px", flex: 1, display: "flex", flexDirection: "column", gap: 20, maxWidth: 720 }}>
          <h1 style={{ margin: 0, fontSize: "clamp(40px,6vw,72px)", fontWeight: 400, letterSpacing: "-.045em", lineHeight: 1 }}>{t("a_pick")}</h1>
          <p className="cap" style={{ margin: 0 }}>{t("a_drop")}</p>
          <input ref={input} type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff" hidden onChange={(e) => { setFile(e.target.files?.[0] ?? null); reset(); }} />
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <button type="button" className="pill o" onClick={() => input.current?.click()}>{t("a_pick")}</button>
            {file && <span className="cap">{file.name} ({Math.round(file.size / 1024)} KB)</span>}
          </div>
          <div><button type="button" className="pill k" disabled={!file || state.status === "busy"} onClick={() => file && run(file)} style={{ minHeight: 48, opacity: !file || state.status === "busy" ? 0.5 : 1 }}>{state.status === "busy" ? t("a_busy") : t("a_analyze")}</button></div>
          {errText && <div role="alert" style={{ borderRadius: 20, background: "var(--errbg)", color: "var(--errfg)", padding: "16px 20px", fontSize: 14 }}>{errText}</div>}
          <div className="warn"><strong style={{ fontWeight: 600 }}>{t("disc")}</strong></div>
        </main>
      )}
      <div style={{ padding: "18px 32px", fontSize: 12, color: "var(--t3)", display: "flex", gap: "6px 24px", flexWrap: "wrap" }}>
        <span>{t("a_foot")}</span><span>20 MB</span><span>ResNet-50 (heads.pt)</span><span>DIBaS (32 Species)</span><span style={{ color: "var(--ink)", fontWeight: 500 }}>{t("disc")}</span>
      </div>
      <Footer />
    </div>
  );
}
