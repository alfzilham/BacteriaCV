/** Kontrak /api/predict. Tidak boleh berubah tanpa persetujuan owner. */
export type Prediction = {
  shape_label: string;
  shape_confidence: number;
  gram_label: string;
  gram_confidence: number;
  shape_level: string;
  gram_level: string;
  segmentation_ok: boolean;
};

export type PredictResponse = {
  prediction: Prediction;
  stage_index: number;
  stage_names: string[];
  panels: string[];
  notes_text: string[];
  disclaimer: string;
  failed_stages: number[];
};

export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
export const ACCEPTED_EXT = [".png", ".jpg", ".jpeg", ".tif", ".tiff"];

export const apiBase = (): string => (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export const checkFile = (name: string, size: number): "ok" | "type" | "size" => {
  const lower = name.toLowerCase();
  if (!ACCEPTED_EXT.some((e) => lower.endsWith(e))) return "type";
  return size > MAX_UPLOAD_BYTES ? "size" : "ok";
};
