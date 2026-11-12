import { createHmac, timingSafeEqual } from "node:crypto";
import { revalidateTag } from "next/cache";

export const runtime = "nodejs";

const valid = (secret: string, body: string, header: string | null): boolean => {
  if (!header?.startsWith("sha256=")) return false;
  const expected = Buffer.from(createHmac("sha256", secret).update(body).digest("hex"));
  const given = Buffer.from(header.slice(7));
  return expected.length === given.length && timingSafeEqual(expected, given);
};

/** Webhook GitHub Release: setiap rilis baru menyegarkan daftar changelog. */
export async function POST(req: Request): Promise<Response> {
  const secret = process.env.GITHUB_WEBHOOK_SECRET;
  if (!secret) return new Response("webhook belum dikonfigurasi", { status: 503 });
  const body = await req.text();
  if (!valid(secret, body, req.headers.get("x-hub-signature-256"))) return new Response("tanda tangan salah", { status: 401 });
  if (req.headers.get("x-github-event") === "release") revalidateTag("releases");
  return new Response("ok");
}
