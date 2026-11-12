"use client";

import { useCallback, useState } from "react";
import { demoJsonUrl } from "@/models/demo";
import { apiBase, checkFile, type PredictResponse } from "@/models/prediction";

export type AnalyzeState =
  | { status: "idle" }
  | { status: "busy" }
  | { status: "error"; reason: "type" | "size" | "server" | "network" | "demo"; detail?: string }
  | { status: "done"; data: PredictResponse; demoId?: string };

export function useAnalyze() {
  const [state, setState] = useState<AnalyzeState>({ status: "idle" });

  const run = useCallback(async (file: File) => {
    const check = checkFile(file.name, file.size);
    if (check !== "ok") { setState({ status: "error", reason: check }); return; }
    setState({ status: "busy" });
    const body = new FormData();
    body.append("file", file);
    try {
      const res = await fetch(`${apiBase()}/api/predict?stage=0`, { method: "POST", body });
      if (!res.ok) {
        const j = (await res.json().catch(() => ({}))) as { detail?: string };
        setState({ status: "error", reason: "server", detail: j.detail });
        return;
      }
      setState({ status: "done", data: (await res.json()) as PredictResponse });
    } catch {
      setState({ status: "error", reason: "network" });
    }
  }, []);

  /** Memuat hasil bawaan dari public/demo, tanpa memanggil API. */
  const runDemo = useCallback(async (id: string) => {
    setState({ status: "busy" });
    try {
      const res = await fetch(demoJsonUrl(id));
      if (!res.ok) { setState({ status: "error", reason: "demo" }); return; }
      const json = (await res.json()) as PredictResponse;
      setState({ status: "done", data: { ...json, panels: [] }, demoId: id });
    } catch {
      setState({ status: "error", reason: "demo" });
    }
  }, []);

  const reset = useCallback(() => setState({ status: "idle" }), []);
  return { state, run, runDemo, reset };
}
