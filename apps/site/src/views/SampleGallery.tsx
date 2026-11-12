"use client";

import { usePrefs } from "@/controllers/prefs";
import { DEMO_SAMPLES } from "@/models/demo";

export function SampleGallery({ onPick, disabled }: { onPick: (id: string) => void; disabled: boolean }) {
  const { t } = usePrefs();
  return (
    <section aria-labelledby="dm-h">
      <h2 id="dm-h" className="h3">{t("dm_h")}</h2>
      <p className="cap" style={{ margin: "0 0 16px" }}>{t("dm_d")}</p>
      <div className="sg">
        {DEMO_SAMPLES.map((s) => (
          <button key={s.id} type="button" className="sgc" disabled={disabled} onClick={() => onPick(s.id)}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={s.image} alt="" width={320} height={240} loading="lazy" />
            <span>{t(s.nameKey)}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
