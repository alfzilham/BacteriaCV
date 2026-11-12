"use client";

import { useEffect, useState } from "react";
import { apiBase } from "@/models/prediction";

export type ApiHealth = "checking" | "online" | "offline";
const TIMEOUT_MS = 4000;

/** Memeriksa /api/health sekali saat halaman dibuka. Offline berarti halaman memakai mode demo. */
export function useApiHealth(): ApiHealth {
  const [health, setHealth] = useState<ApiHealth>("checking");
  useEffect(() => {
    const ctl = new AbortController();
    const timer = window.setTimeout(() => ctl.abort(), TIMEOUT_MS);
    fetch(`${apiBase()}/api/health`, { signal: ctl.signal })
      .then(async (res) => {
        const body = (await res.json().catch(() => ({}))) as { status?: string };
        setHealth(res.ok && body.status === "ready" ? "online" : "offline");
      })
      .catch(() => setHealth("offline"))
      .finally(() => window.clearTimeout(timer));
    return () => {
      window.clearTimeout(timer);
      ctl.abort();
    };
  }, []);
  return health;
}
