import type { CopyKey } from "./copy";

/** Hasil demo bawaan, dibuat sekali oleh tools/export-stages.ps1 ke public/demo. */
export const DEMO_DIR = "/demo";

export type DemoSample = { id: string; image: string; nameKey: CopyKey };

export const DEMO_SAMPLES: readonly DemoSample[] = [
  { id: "contoh-1-bacilli-positif", image: "/samples/contoh-1-bacilli-positif.png", nameKey: "dm_s1" },
  { id: "contoh-2-bacilli-negatif", image: "/samples/contoh-2-bacilli-negatif.png", nameKey: "dm_s2" },
  { id: "contoh-3-cocci-positif", image: "/samples/contoh-3-cocci-positif.png", nameKey: "dm_s3" },
  { id: "contoh-4-cocci-negatif", image: "/samples/contoh-4-cocci-negatif.png", nameKey: "dm_s4" },
];

export const demoJsonUrl = (id: string): string => `${DEMO_DIR}/${id}.json`;
export const demoPanelUrl = (id: string, stage: number): string => `${DEMO_DIR}/${id}-stage-${stage + 1}.png`;
