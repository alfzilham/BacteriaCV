"use client";

/** Pita berjalan tanpa henti. Berhenti saat disorot, dan menjadi daftar biasa bila gerak dikurangi. */
export function Marquee({ items, label }: { items: readonly string[]; label: string }) {
  return (
    <div className="mq" role="region" aria-label={label}>
      <div className="mq-track">
        {[0, 1].map((copy) => (
          <ul key={copy} className="mq-set" aria-hidden={copy === 1 ? true : undefined}>
            {items.map((item) => <li key={item} className="tag">{item}</li>)}
          </ul>
        ))}
      </div>
    </div>
  );
}
