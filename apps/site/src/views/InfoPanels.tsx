"use client";

import { usePrefs } from "@/controllers/prefs";
import type { HistoryEntry } from "@/controllers/history";

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

type Props = { history: HistoryEntry[]; onOpenDemo: (id: string) => void; busy: boolean };

/** Legenda keyakinan, tips gambar, dan riwayat sesi. Dipakai di keadaan awal dan setelah hasil. */
export function InfoPanels({ history, onOpenDemo, busy }: Props) {
  const { t } = usePrefs();
  const levels = [["lg_hi", 3], ["lg_me", 2], ["lg_lo", 1]] as const;
  return (
    <div className="cards" style={{ padding: "0 24px" }}>
      <section className="tile" aria-labelledby="lg-h">
        <h2 id="lg-h" className="h3">{t("lg_h")}</h2>
        <ul className="plain">
          {levels.map(([k, n]) => (
            <li key={k} style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span aria-hidden="true" style={{ display: "flex", gap: 3 }}>
                {[1, 2, 3].map((i) => <i key={i} style={{ width: 7, height: 7, borderRadius: "50%", background: i <= n ? "var(--ink)" : "var(--dot)" }} />)}
              </span>
              {t(k)}
            </li>
          ))}
        </ul>
        <p className="cap" style={{ margin: "12px 0 0" }}>{t("lg_d")}</p>
      </section>
      <section className="tile" aria-labelledby="tp-h">
        <h2 id="tp-h" className="h3">{t("tp_h")}</h2>
        <ul className="plain">
          {(["tp_1", "tp_2", "tp_3"] as const).map((k) => <li key={k}>{t(k)}</li>)}
        </ul>
      </section>
      <section className="tile" aria-labelledby="hs-h">
        <h2 id="hs-h" className="h3">{t("hs_h")}</h2>
        {history.length === 0 ? (
          <p className="cap" style={{ margin: 0 }}>{t("hs_e")}</p>
        ) : (
          <ul className="plain">
            {history.map((h) => (
              <li key={h.at} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
                <span>{cap(h.shape)}, Gram {h.gram}</span>
                {h.demoId && <button type="button" className="chip" disabled={busy} onClick={() => onOpenDemo(h.demoId as string)}>{t("hs_o")}</button>}
              </li>
            ))}
          </ul>
        )}
        <p className="cap" style={{ margin: "12px 0 0" }}>{t("hs_n")}</p>
      </section>
    </div>
  );
}
