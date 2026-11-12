"use client";

import { usePrefs } from "@/controllers/prefs";
import { DocsShell } from "./DocsShell";
import { Faq } from "./Faq";

export function DocsFaq() {
  const { t } = usePrefs();
  return (
    <DocsShell page="faq">
      <h1>{t("faq_h")}</h1>
      <Faq />
    </DocsShell>
  );
}
