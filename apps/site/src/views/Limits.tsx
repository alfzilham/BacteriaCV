"use client";

import { usePrefs } from "@/controllers/prefs";
import { DocList } from "./DocBits";
import { DocsShell } from "./DocsShell";

export function Limits() {
  const { t } = usePrefs();
  return (
    <DocsShell page="limits">
      <h1>{t("lm_h")}</h1>
      <p>{t("lm_d")}</p>
      <div className="warn"><strong style={{ fontWeight: 600 }}>{t("disc")}</strong> {t("lim_seg")}</div>
      <DocList items={[t("lm_seg"), t("lm_test"), t("lm_spiral"), t("lm_scope"), t("lm_img"), t("qa3_a"), t("lm_host"), t("lm_and")]} />
    </DocsShell>
  );
}
