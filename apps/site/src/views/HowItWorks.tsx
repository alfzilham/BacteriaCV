"use client";

import { usePrefs } from "@/controllers/prefs";
import { DocList } from "./DocBits";
import { DocsShell } from "./DocsShell";

const STAGES = [["st1", "sd1"], ["st2", "sd2"], ["st3", "sd3"], ["st4", "sd4"], ["st5", "sd5"]] as const;

export function HowItWorks() {
  const { t } = usePrefs();
  return (
    <DocsShell page="how-it-works">
      <h1>{t("hw_h")}</h1>
      <p>{t("hw_d")}</p>
      <h2>{t("hw_st_h")}</h2>
      <DocList ordered items={STAGES.map(([title, desc]) => <span key={title}><strong style={{ fontWeight: 600, color: "var(--ink)" }}>{t(title)}.</strong> {t(desc)}</span>)} />
      <p>{t("hw_seg")}</p>
      <h2>{t("hw_m_h")}</h2>
      <DocList items={[t("hw_m1"), t("hw_m2"), t("hw_m3")]} />
      <h2>{t("d_conf_h")}</h2>
      <DocList items={[t("d_conf_hi"), t("d_conf_me"), t("d_conf_lo")]} />
      <h2>{t("hw_d_h")}</h2>
      <DocList items={[t("hw_d1"), t("hw_d2")]} />
      <h2>{t("hw_o_h")}</h2>
      <p>{t("hw_o_d")}</p>
      <div className="warn"><strong style={{ fontWeight: 600 }}>{t("disc")}</strong> {t("lim_seg")}</div>
    </DocsShell>
  );
}
