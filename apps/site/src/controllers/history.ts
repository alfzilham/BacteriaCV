"use client";

import { useCallback, useEffect, useState } from "react";

export const HISTORY_KEY = "bcv-history";
export const HISTORY_MAX = 5;

/** Hanya label yang disimpan. Gambar unggahan tidak pernah disimpan di peramban. */
export type HistoryEntry = { at: number; shape: string; gram: string; demoId?: string };

const isEntry = (e: unknown): e is HistoryEntry =>
  typeof e === "object" && e !== null &&
  typeof (e as HistoryEntry).at === "number" &&
  typeof (e as HistoryEntry).shape === "string" &&
  typeof (e as HistoryEntry).gram === "string";

const read = (): HistoryEntry[] => {
  try {
    const raw = sessionStorage.getItem(HISTORY_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter(isEntry).slice(0, HISTORY_MAX) : [];
  } catch {
    return [];
  }
};
const write = (items: HistoryEntry[]) => {
  try { sessionStorage.setItem(HISTORY_KEY, JSON.stringify(items)); } catch { /* penyimpanan dapat diblokir */ }
};

export function useHistory() {
  const [items, setItems] = useState<HistoryEntry[]>([]);
  useEffect(() => { setItems(read()); }, []);
  const add = useCallback((entry: HistoryEntry) => {
    const next = [entry, ...read()].slice(0, HISTORY_MAX);
    write(next);
    setItems(next);
  }, []);
  return { items, add };
}
