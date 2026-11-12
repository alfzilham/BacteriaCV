"use client";

import { usePrefs } from "@/controllers/prefs";
import type { CopyKey } from "@/models/copy";

type Channel = { title: CopyKey; desc: CopyKey; value: string; href: string };

export const CHANNELS: readonly Channel[] = [
  { title: "ct_is_t", desc: "ct_is_d", value: "alfzilham/BacteriaCV/issues", href: "https://github.com/alfzilham/BacteriaCV/issues" },
  { title: "ct_ds_t", desc: "ct_ds_d", value: "alfzilham/BacteriaCV/discussions", href: "https://github.com/alfzilham/BacteriaCV/discussions" },
  { title: "ct_gh_t", desc: "ct_gh_d", value: "github.com/alfzilham", href: "https://github.com/alfzilham" },
  { title: "ct_in_t", desc: "ct_in_d", value: "linkedin.com/in/alfiz-ilham093a", href: "https://www.linkedin.com/in/alfiz-ilham093a/" },
  { title: "ct_em_t", desc: "ct_em_d", value: "alfizilham@gmail.com", href: "mailto:alfizilham@gmail.com" },
  { title: "ct_wa_t", desc: "ct_wa_d", value: "+62 878 2206 5257", href: "https://wa.me/6287822065257" },
];

export function Contact() {
  const { t } = usePrefs();
  return (
    <div className="cards">
      {CHANNELS.map((c) => {
        const external = c.href.startsWith("http");
        return (
          <a key={c.title} className="tile hv ct" href={c.href} {...(external ? { target: "_blank", rel: "noreferrer" } : {})}>
            <span style={{ fontSize: 24, letterSpacing: "-.03em" }}>{t(c.title)}</span>
            <span className="cap">{t(c.desc)}</span>
            <span style={{ marginTop: "auto", fontSize: 14, fontWeight: 500, overflowWrap: "anywhere" }}>{c.value}</span>
          </a>
        );
      })}
    </div>
  );
}
