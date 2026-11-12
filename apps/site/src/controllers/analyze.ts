"use client";

import { useCallback, useState } from "react";
import { apiBase, checkFile, type PredictResponse } from "@/models/prediction";

export type AnalyzeState =
  | { status: "idle" }
  | { status: "busy" }
  | { status: "error"; reason: "type" | "size" | "server" | "network"; detail?: string }
  | { status: "done"; data: PredictResponse };

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

  const reset = useCallback(() => setState({ status: "idle" }), []);
  return { state, run, reset };
}
