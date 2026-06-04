"use client";

import { useEffect, useState } from "react";
import { LoaderCircle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

type CSCfg = { url: string; apiKey: string };

// Browser profiles that are NOT real Google-account sessions (system / tool
// profiles) — hidden from the reuse picker so users only see real accounts.
const SYSTEM_PROFILES = new Set([
  "default", "nopecha", "phatnguoi", "phatnguoi-manual",
  "stealth-check", "cfdemo",
]);

// Keep only profiles that look like a real onboarded Google account, dropping
// junk/test/probe dirs and invalid names (e.g. "chatgpt-*", names with a comma
// from a stray account-type label, probes, test-onboard-*, *-default).
function isAccountProfile(n: string): boolean {
  if (!n) return false;
  if (/[,*\s]/.test(n)) return false; // junk: comma / asterisk / whitespace
  if (SYSTEM_PROFILES.has(n)) return false;
  if (/(^|[-_])default$/i.test(n)) return false; // default, gemini-web-default
  if (/^diag\d*$/i.test(n)) return false;
  if (/^pn-/i.test(n)) return false;
  if (/(^|-)probe\d*$/i.test(n)) return false; // chatgpt-probe, -probe2
  if (/^test[-_]|[-_]test$|^nonexistent/i.test(n)) return false; // test-*, *-test
  return true;
}

/**
 * Shared "reuse an already-onboarded profile" control.
 *
 * Lists the captcha-solver browser profiles (each = one Google account that
 * already has a live session) and lets the user pick one + click "Tái dùng".
 * The parent card supplies `onReuse(profile)` with its own provider-specific
 * logic (ChatGPT → add token to pool, Gemini → save config, Flow → add
 * account). This powers "Cách A": log in any provider first, then add the
 * others on the SAME profile without re-entering email/password.
 */
export function ReuseProfilePicker({
  cs,
  onReuse,
}: {
  cs: CSCfg;
  onReuse: (profile: string) => Promise<void>;
}) {
  const [profiles, setProfiles] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);

  async function load() {
    if (!cs.url) return;
    setLoading(true);
    try {
      const res = await fetch(`${cs.url}/v1/session/list`, {
        headers: { Authorization: `Bearer ${cs.apiKey}` },
      });
      const data = await res.json();
      const names: string[] = (data.profiles || [])
        .map((p: { name?: string }) => p.name || "")
        .filter(isAccountProfile)
        .sort();
      setProfiles(names);
      setSelected((s) => (s && names.includes(s) ? s : names[0] || ""));
    } catch {
      /* network blip — leave list as-is */
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cs.url, cs.apiKey]);

  return (
    <div className="flex items-center gap-2">
      <select
        className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
        value={selected}
        onChange={(e) => setSelected(e.target.value)}
        disabled={busy || loading}
      >
        {profiles.length === 0 && (
          <option value="">{loading ? "Đang tải…" : "(chưa có profile)"}</option>
        )}
        {profiles.map((p) => (
          <option key={p} value={p}>
            {p}
          </option>
        ))}
      </select>
      <Button
        type="button"
        variant="outline"
        size="icon"
        onClick={() => void load()}
        disabled={busy || loading}
        title="Tải lại danh sách profile"
      >
        <RefreshCw className="h-4 w-4" />
      </Button>
      <Button
        type="button"
        onClick={async () => {
          if (!selected) return;
          setBusy(true);
          try {
            await onReuse(selected);
          } finally {
            setBusy(false);
          }
        }}
        disabled={busy || loading || !selected}
      >
        {busy ? <LoaderCircle className="mr-1 h-4 w-4 animate-spin" /> : null}
        Tái dùng
      </Button>
    </div>
  );
}
