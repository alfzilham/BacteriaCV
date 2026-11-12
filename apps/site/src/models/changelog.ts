export type Platform = "web" | "android";
export type Release = { tag: string; platform: Platform; date: string; title: string; notes: string };

/** Pemetaan tag rilis ke platform. web-v... untuk web, mobile-v... untuk Android. */
export const platformOfTag = (tag: string): Platform | null =>
  tag.startsWith("web-v") ? "web" : tag.startsWith("mobile-v") ? "android" : null;

/** Entri placeholder dipakai sampai rilis pertama ada di GitHub. */
export const PLACEHOLDERS: Release[] = [
  { tag: "web-v[x.y.z]", platform: "web", date: "[date]", title: "[release title]", notes: "[what changed, taken from the GitHub Release notes]" },
  { tag: "mobile-v[x.y.z]", platform: "android", date: "[date]", title: "[release title]", notes: "[what changed, taken from the GitHub Release notes]" },
];

type GhRelease = { tag_name: string; name: string | null; body: string | null; published_at: string | null; draft: boolean };

export const toReleases = (raw: GhRelease[]): Release[] =>
  raw.flatMap((r) => {
    const platform = platformOfTag(r.tag_name);
    if (!platform || r.draft) return [];
    return [{ tag: r.tag_name, platform, date: (r.published_at ?? "").slice(0, 10), title: r.name ?? r.tag_name, notes: r.body ?? "" }];
  });
