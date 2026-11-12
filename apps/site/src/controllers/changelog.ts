import { PLACEHOLDERS, toReleases, type Release } from "@/models/changelog";

/** Ambil rilis dari GitHub. Bila gagal atau kosong, pakai placeholder desain. */
export async function loadReleases(): Promise<Release[]> {
  const repo = process.env.GITHUB_REPO ?? "alfzilham/BacteriaCV";
  const headers: Record<string, string> = { Accept: "application/vnd.github+json" };
  if (process.env.GITHUB_TOKEN) headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  try {
    const res = await fetch(`https://api.github.com/repos/${repo}/releases?per_page=50`, { headers, next: { revalidate: 3600, tags: ["releases"] } });
    if (!res.ok) return PLACEHOLDERS;
    const list = toReleases(await res.json());
    return list.length ? list : PLACEHOLDERS;
  } catch {
    return PLACEHOLDERS;
  }
}
