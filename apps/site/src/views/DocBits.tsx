import type { ReactNode } from "react";

/** Daftar bernomor atau berpoin untuk halaman dokumentasi. */
export function DocList({ items, ordered = false }: { items: ReactNode[]; ordered?: boolean }) {
  const Tag = ordered ? "ol" : "ul";
  return (
    <Tag className="dl">
      {items.map((item, i) => <li key={i}>{item}</li>)}
    </Tag>
  );
}

/** Blok kode. Isinya literal, tidak diterjemahkan. */
export function Code({ children }: { children: string }) {
  return <pre><code>{children}</code></pre>;
}

/** Nama berkas, variabel, atau perintah di dalam kalimat. */
export function Ic({ children }: { children: ReactNode }) {
  return <code className="ic">{children}</code>;
}
