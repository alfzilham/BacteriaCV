import { loadReleases } from "@/controllers/changelog";
import { ChangelogView } from "@/views/ChangelogView";

export default async function Page() {
  return <ChangelogView releases={await loadReleases()} />;
}
