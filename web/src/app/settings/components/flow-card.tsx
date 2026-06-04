"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { LoaderCircle, Plus, Save, Trash2, ExternalLink, Sparkles, KeyRound, RotateCw, Smartphone, X, Shield, Eye, EyeOff } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { request } from "@/lib/request";
import { SavedAccountsSelect } from "@/components/saved-accounts-select";
import { ReuseProfilePicker } from "./reuse-profile-picker";
import { generateTotpCode, totpSecondsRemaining } from "@/lib/totp";

type FlowAccount = {
  profile: string;
  project_id: string;
  label?: string;
};

type AutoLoginState = {
  profile: string;
  email: string;
  state: "none" | "pending" | "starting" | "running" | "need_tap" | "need_code" | "success" | "failed";
  message: string;
  tap_number?: string | null;
  elapsed_sec?: number;
  error?: string | null;
};

type FlowConfig = {
  enabled: boolean;
  captcha_solver_url: string;
  captcha_solver_api_key: string;
  accounts: FlowAccount[];
  cooldown_seconds?: number;
};

const DEFAULT_BASE_PROFILE = "google-fx";
const EMPTY_ACCOUNT: FlowAccount = { profile: DEFAULT_BASE_PROFILE, project_id: "", label: "Main" };

/** Find the next unused suffix for a base profile name.
 *
 *   existing: ["google-fx"]                        → "google-fx-1"
 *   existing: ["google-fx", "google-fx-1"]         → "google-fx-2"
 *   existing: ["google-fx-1", "google-fx-3"]       → "google-fx" (base free)
 *   existing: ["google-fx", "google-fx-1", "google-fx-2"] → "google-fx-3"
 *   existing: []                                   → "google-fx"
 */
function nextProfileName(existing: string[], base = DEFAULT_BASE_PROFILE): string {
  const set = new Set(existing);
  if (!set.has(base)) return base;
  for (let i = 1; i < 1000; i++) {
    const candidate = `${base}-${i}`;
    if (!set.has(candidate)) return candidate;
  }
  return `${base}-${Date.now()}`;
}

/** Suggest the next label that fits the FIFO fallback chain. Order:
 *  Main → Backup → Spare 1 → Spare 2 → Spare 3 → Standby → Spare 4 ...
 *  Skips labels already in use so two accounts never collide.
 */
function nextLabel(existing: string[]): string {
  const used = new Set(existing.map((s) => s.trim()).filter(Boolean));
  const preset = ["Main", "Backup", "Spare 1", "Spare 2", "Spare 3", "Standby"];
  for (const label of preset) {
    if (!used.has(label)) return label;
  }
  // Pool past the preset list — keep generating Spare N.
  for (let i = 4; i < 1000; i++) {
    const candidate = `Spare ${i}`;
    if (!used.has(candidate)) return candidate;
  }
  return `Account ${used.size + 1}`;
}

export function FlowCard() {
  const [cfg, setCfg] = useState<FlowConfig>({
    enabled: true,
    captcha_solver_url: "http://172.16.10.38:8010",
    captcha_solver_api_key: "",
    accounts: [],
    cooldown_seconds: 3600,
  });
  const [draft, setDraft] = useState<FlowAccount>({ ...EMPTY_ACCOUNT });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  // Track which fields the user has manually edited so we don't
  // overwrite custom entries when accounts change. Sticky once typed.
  const [manuallyEditedProfile, setManuallyEditedProfile] = useState(false);
  const [manuallyEditedLabel, setManuallyEditedLabel] = useState(false);

  // Auto-login state
  const [autoLogin, setAutoLogin] = useState<{ email: string; password: string; code: string; totpSecret: string }>({
    email: "",
    password: "",
    code: "",
    totpSecret: "",
  });
  const [selectedAccount, setSelectedAccount] = useState("");
  const [isSavingAccount, setIsSavingAccount] = useState(false);
  const [savedRefreshKey, setSavedRefreshKey] = useState(0);
  const [loginSession, setLoginSession] = useState<AutoLoginState | null>(null);
  const [totpCode, setTotpCode] = useState("");
  const [totpRemaining, setTotpRemaining] = useState(30);
  const pollIntervalRef = useRef<number | null>(null);
  const totpTimerRef = useRef<number | null>(null);
  const [showPassword, setShowPassword] = useState(true);

  // Auto-refresh TOTP code
  useEffect(() => {
    if (!autoLogin.totpSecret.trim()) { setTotpCode(""); return; }
    const refresh = async () => {
      try {
        setTotpCode(await generateTotpCode(autoLogin.totpSecret));
        setTotpRemaining(totpSecondsRemaining());
      } catch { setTotpCode(""); }
    };
    void refresh();
    totpTimerRef.current = window.setInterval(refresh, 5000);
    return () => { if (totpTimerRef.current) window.clearInterval(totpTimerRef.current); };
  }, [autoLogin.totpSecret]);

  useEffect(() => { fetchCfg(); }, []);

  // Cleanup poll on unmount
  useEffect(() => () => {
    if (pollIntervalRef.current) {
      window.clearInterval(pollIntervalRef.current);
    }
  }, []);

  // Suggested values based on what's already in the pool. Both
  // re-compute whenever cfg.accounts changes (after add/remove/save).
  const suggestedProfile = useMemo(
    () => nextProfileName(cfg.accounts.map((a) => a.profile)),
    [cfg.accounts]
  );
  const suggestedLabel = useMemo(
    () => nextLabel(cfg.accounts.map((a) => a.label || "")),
    [cfg.accounts]
  );

  // Auto-fill draft with the suggestion unless the user has typed their
  // own. Triggers on every account-list update.
  useEffect(() => {
    setDraft((d) => ({
      ...d,
      profile: manuallyEditedProfile ? d.profile : suggestedProfile,
      label:   manuallyEditedLabel   ? d.label   : suggestedLabel,
    }));
  }, [suggestedProfile, suggestedLabel, manuallyEditedProfile, manuallyEditedLabel]);

  async function fetchCfg() {
    setLoading(true);
    try {
      const data = await request.get("/api/settings");
      const flow = ((data.data as any)?.config?.providers || {}).flow || {};
      setCfg({
        enabled: flow.enabled !== false,
        captcha_solver_url: flow.captcha_solver_url || "http://172.16.10.38:8010",
        captcha_solver_api_key: flow.captcha_solver_api_key || "",
        accounts: Array.isArray(flow.accounts) ? flow.accounts : [],
        cooldown_seconds: typeof flow.cooldown_seconds === "number" ? flow.cooldown_seconds : 3600,
      });
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }

  async function save(next: FlowConfig) {
    setSaving(true);
    try {
      // /api/settings does a shallow merge at the top level — wrapping
      // the payload as `{ config: { providers: { flow: next } } }` would
      // create a literal `config` key in settings and leave `providers`
      // untouched (so deletes/edits silently no-op). Send `providers` at
      // the top level, and merge into the existing providers dict so we
      // don't wipe sibling providers (gemini_web, chatgpt_web, ...).
      const cur = await request.get("/api/settings");
      const providers = { ...(((cur.data as any)?.config?.providers) || {}) };
      providers.flow = next;
      await request.post("/api/settings", { providers });
      toast.success("Đã lưu cấu hình Flow");
      setCfg(next);
    } catch (e: any) {
      toast.error(e?.message || "Lỗi lưu");
    } finally { setSaving(false); }
  }

  function addAccount() {
    if (!draft.profile.trim() || !draft.project_id.trim()) {
      toast.error("Profile + project_id là bắt buộc");
      return;
    }
    const next = { ...cfg, accounts: [...cfg.accounts, { ...draft, label: draft.label?.trim() || draft.profile }] };
    void save(next);
    // Reset draft and clear manual-edit flags so the useEffect re-suggests
    // the next available profile + label after the save completes.
    setDraft({ ...EMPTY_ACCOUNT });
    setManuallyEditedProfile(false);
    setManuallyEditedLabel(false);
  }

  function removeAccount(idx: number) {
    const next = { ...cfg, accounts: cfg.accounts.filter((_, i) => i !== idx) };
    void save(next);
  }

  // Cách A — reuse an existing profile's Google session for Flow: fetch/create
  // a project on that profile (no login) and add it to the pool.
  async function reuseAccount(prof: string) {
    if (cfg.accounts.some((a) => a.profile === prof)) {
      toast.info(`${prof} đã có trong pool`);
      return;
    }
    const url = cfg.captcha_solver_url;
    const key = cfg.captcha_solver_api_key;
    try {
      toast.info(`Đang lấy Flow project cho ${prof}…`);
      const res = await fetch(`${url}/v1/google/flow/get-or-create-project`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${key}`, "Content-Type": "application/json" },
        body: JSON.stringify({ profile: prof, headless: true, timeout: 150 }),
      });
      if (!res.ok) throw new Error(`get-project HTTP ${res.status}`);
      const data = await res.json();
      const projectId = data.project_id;
      if (!projectId) throw new Error(data.detail || data.error || "no project_id");
      const label = nextLabel(cfg.accounts.map((a) => a.label || ""));
      const next = { ...cfg, accounts: [...cfg.accounts, { profile: prof, project_id: String(projectId), label }] };
      await save(next);
      toast.success(`Đã thêm ${prof} vào Flow (project ${String(projectId).slice(0, 8)}…)`);
    } catch (e: any) {
      toast.error(`Reuse Flow lỗi: ${e?.message || e}`);
    }
  }

  function openNoVNC() {
    if (!cfg.captcha_solver_url) {
      toast.error("Cần điền captcha_solver_url trước");
      return;
    }
    const novncUrl = cfg.captcha_solver_url.replace(":8010", ":6080") + "/vnc.html?autoconnect=1";
    window.open(novncUrl, "_blank");
  }

  async function triggerManualLogin(force = false) {
    if (!draft.profile.trim()) {
      toast.error("Cần điền profile trước");
      return;
    }
    try {
      const res = await fetch(`${cfg.captcha_solver_url}/v1/session/manual-login`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${cfg.captcha_solver_api_key}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          profile: draft.profile.trim(),
          url: "https://labs.google/fx/vi/tools/flow",
          force,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      toast.success(force ? "Đã khởi động lại Chrome — mở noVNC" : "Đã mở browser session — mở noVNC để login Google");
      openNoVNC();
    } catch (e: any) {
      toast.error(`Lỗi gọi manual-login: ${e?.message}`);
    }
  }

  function stopPolling() {
    if (pollIntervalRef.current) {
      window.clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }

  async function pollLoginStatus(profile: string, onSuccess?: (s: AutoLoginState) => void) {
    try {
      const res = await fetch(
        `${cfg.captcha_solver_url}/v1/session/${encodeURIComponent(profile)}/auto-login-status`,
        { headers: { Authorization: `Bearer ${cfg.captcha_solver_api_key}` } },
      );
      if (!res.ok) return;
      const data = await res.json();
      setLoginSession(data);
      if (data.state === "success" || data.state === "failed") {
        stopPolling();
        if (data.state === "success") {
          toast.success("Đăng nhập thành công 🎉");
          if (onSuccess) onSuccess(data);
        } else {
          toast.error(`Auto-login lỗi: ${data.error || data.message}`);
        }
      }
    } catch {
      /* network blip — keep polling */
    }
  }

  // ── 1-click full automation ──
  // Auto-login → wait for success → call /v1/google/flow/get-or-create-project
  // → push the {profile, project_id, label} into the Flow pool config.
  // Handles 2FA prompts the same way as startAutoLogin (UI shows tap-match
  // number / SMS code input), and stops at any failure with a toast.
  const [oneClickRunning, setOneClickRunning] = useState(false);
  const [oneClickStep, setOneClickStep] = useState<string>("");

  async function oneClickAddAccount() {
    if (!autoLogin.email.trim() || !autoLogin.password) {
      toast.error("Cần điền email + mật khẩu cho 1-click");
      return;
    }
    // Account-centric profile (provider-neutral) so logging in via Flow vs
    // ChatGPT vs Gemini produces the SAME profile for one Google account —
    // one profile per account, clean cross-provider reuse. (Was google-fx-N.)
    const local = (autoLogin.email.split("@")[0] || "fx").replace(/[^a-z0-9-]/gi, "-");
    const profile = `google-${local}`;
    const label = suggestedLabel;
    stopPolling();
    setOneClickRunning(true);
    setOneClickStep("Đang đăng nhập Google...");
    try {
      // 1) Start auto-login
      const loginRes = await fetch(`${cfg.captcha_solver_url}/v1/session/auto-login`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${cfg.captcha_solver_api_key}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          profile,
          email: autoLogin.email.trim(),
          password: autoLogin.password,
          totp_secret: autoLogin.totpSecret.trim(),
          prefer_method: "auth",
        }),
      });
      if (!loginRes.ok) throw new Error(`auto-login HTTP ${loginRes.status}`);
      const initialSession = await loginRes.json();
      setLoginSession(initialSession);
      openNoVNC();

      // 2) Poll for success (or 2FA prompt). User must complete 2FA via
      // the existing tap/code UI (panel below). We continue when state
      // becomes success.
      const onSuccess = async () => {
        try {
          setOneClickStep("Đăng nhập OK — đang lấy/tạo Flow project...");
          // 3) Get or create project
          const projRes = await fetch(`${cfg.captcha_solver_url}/v1/google/flow/get-or-create-project`, {
            method: "POST",
            headers: {
              "Authorization": `Bearer ${cfg.captcha_solver_api_key}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ profile, headless: false, timeout: 90 }),
          });
          if (!projRes.ok) {
            const err = await projRes.json().catch(() => ({}));
            throw new Error(err.detail || `get-or-create-project HTTP ${projRes.status}`);
          }
          const proj = await projRes.json();
          setOneClickStep(`Got project ${proj.project_id.slice(0, 8)}... (${proj.action}) — đang thêm vào pool...`);

          // 4) Save to pool config
          const newAccount: FlowAccount = {
            profile,
            project_id: proj.project_id,
            label,
          };
          const next = { ...cfg, accounts: [...cfg.accounts, newAccount] };
          await save(next);
          setOneClickStep(`Hoàn tất ✅ Account #${next.accounts.length} (${label}) đã sẵn sàng`);
          toast.success(`Đã thêm account ${label} — profile ${profile}`);
          // Clear email/password
          setAutoLogin({ email: "", password: "", code: "", totpSecret: "" });
          setSelectedAccount("");
        } catch (e: any) {
          setOneClickStep("");
          toast.error(`Lỗi sau login: ${e?.message}`);
        } finally {
          setOneClickRunning(false);
        }
      };
      pollIntervalRef.current = window.setInterval(() => {
        void pollLoginStatus(profile, onSuccess);
      }, 1500);
    } catch (e: any) {
      toast.error(`Lỗi 1-click: ${e?.message}`);
      setOneClickRunning(false);
      setOneClickStep("");
    }
  }

  async function startAutoLogin() {
    const profile = draft.profile.trim();
    if (!profile) { toast.error("Cần điền profile trước"); return; }
    if (!autoLogin.email.trim() || !autoLogin.password) {
      toast.error("Cần điền email + mật khẩu");
      return;
    }
    stopPolling();
    try {
      const res = await fetch(`${cfg.captcha_solver_url}/v1/session/auto-login`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${cfg.captcha_solver_api_key}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          profile,
          email: autoLogin.email.trim(),
          password: autoLogin.password,
          totp_secret: autoLogin.totpSecret.trim(),
          prefer_method: "auth",
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setLoginSession(data);
      toast.success("Auto-login đã chạy — theo dõi ở dưới");
      // Open noVNC so user can see Chrome live
      openNoVNC();
      // Start polling every 1.5s
      pollIntervalRef.current = window.setInterval(() => {
        void pollLoginStatus(profile);
      }, 1500);
    } catch (e: any) {
      toast.error(`Lỗi auto-login: ${e?.message}`);
    }
  }

  async function submit2faCode() {
    const profile = loginSession?.profile;
    if (!profile || !autoLogin.code.trim()) {
      toast.error("Cần mã 2FA");
      return;
    }
    try {
      const res = await fetch(
        `${cfg.captcha_solver_url}/v1/session/${encodeURIComponent(profile)}/auto-login-2fa-code`,
        {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${cfg.captcha_solver_api_key}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ code: autoLogin.code.trim() }),
        },
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      toast.success("Đã gửi mã, đợi xác nhận…");
      setAutoLogin((s) => ({ ...s, code: "" }));
    } catch (e: any) {
      toast.error(`Lỗi gửi mã: ${e?.message}`);
    }
  }

  function cancelLoginSession() {
    stopPolling();
    setLoginSession(null);
    setAutoLogin({ email: "", password: "", code: "", totpSecret: "" });
    setSelectedAccount("");
  }

  async function handleSaveAccount() {
    if (!autoLogin.email.trim() || !autoLogin.password) {
      toast.error("Cần email + mật khẩu để lưu");
      return;
    }
    setIsSavingAccount(true);
    try {
      await fetch(`${cfg.captcha_solver_url}/v1/accounts/saved`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${cfg.captcha_solver_api_key}`, "Content-Type": "application/json" },
        body: JSON.stringify({ email: autoLogin.email.trim(), password: autoLogin.password, totp_secret: autoLogin.totpSecret.trim() }),
      });
      toast.success("Đã lưu tài khoản");
      setAutoLogin({ email: "", password: "", code: "", totpSecret: "" });
      setSelectedAccount("");
      setSavedRefreshKey((k) => k + 1);
    } catch {
      toast.error("Lưu tài khoản thất bại");
    } finally {
      setIsSavingAccount(false);
    }
  }

  return (
    <Card className="rounded-3xl border-emerald-100/80 bg-emerald-50/30">
      <CardContent className="space-y-4 p-5">
        {/* Header + global enable */}
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="size-4 text-emerald-600" />
              <h3 className="text-sm font-semibold text-emerald-900">Google Labs Flow</h3>
            </div>
            <p className="text-xs text-emerald-700/70 mt-0.5">
              Sinh ảnh qua labs.google/fx (Nano Banana Pro / 2 / Imagen 4) — chạy qua captcha-solver browser pool
            </p>
          </div>
          <label className="flex items-center gap-2 text-xs text-stone-600">
            <input
              type="checkbox"
              checked={cfg.enabled}
              onChange={(e) => void save({ ...cfg, enabled: e.target.checked })}
              className="size-4 rounded"
            />
            Enabled
          </label>
        </div>

        {/* Captcha-solver connection */}
        <div className="grid gap-3 sm:grid-cols-3 rounded-xl border border-emerald-200/60 bg-white/60 p-3">
          <div className="sm:col-span-2 grid gap-3 sm:grid-cols-2">
            <div>
              <label className="text-xs text-emerald-800">Captcha-solver URL</label>
              <Input
                value={cfg.captcha_solver_url}
                onChange={(e) => setCfg({ ...cfg, captcha_solver_url: e.target.value })}
                onBlur={() => void save(cfg)}
                placeholder="http://172.16.10.38:8010"
                className="mt-1 h-9 rounded-lg border-emerald-200 text-sm font-mono"
              />
            </div>
            <div>
              <label className="text-xs text-emerald-800">Captcha-solver API Key</label>
              <Input
                type="password"
                value={cfg.captcha_solver_api_key}
                onChange={(e) => setCfg({ ...cfg, captcha_solver_api_key: e.target.value })}
                onBlur={() => void save(cfg)}
                placeholder="bearer key"
                className="mt-1 h-9 rounded-lg border-emerald-200 text-sm font-mono"
              />
            </div>
          </div>
          <div>
            <label className="text-xs text-emerald-800">Cooldown sau rate-limit (giây)</label>
            <Input
              type="number"
              min={60}
              max={86400}
              step={60}
              value={cfg.cooldown_seconds ?? 3600}
              onChange={(e) => setCfg({ ...cfg, cooldown_seconds: parseInt(e.target.value) || 3600 })}
              onBlur={() => void save(cfg)}
              placeholder="3600"
              className="mt-1 h-9 rounded-lg border-emerald-200 text-sm font-mono"
            />
            <p className="mt-1 text-[10px] text-emerald-700/70">
              {Math.round((cfg.cooldown_seconds ?? 3600) / 60)} phút · Khi 1 account dính 429/quota → skip trong khoảng này. Auto re-enter pool khi hết.
            </p>
          </div>
        </div>

        {/* Strict-priority fallback explainer */}
        <div className="rounded-xl border border-emerald-200/40 bg-emerald-50/30 px-3 py-2 text-[11px] text-emerald-800/90">
          <span className="font-semibold">Fallback rotation:</span> Main luôn được dùng trước. Khi Main dính quota → tự fallback sang Backup → Spare 1 → Spare 2 → … theo thứ tự trong danh sách. Hết cooldown thì auto re-enter pool ở slot ưu tiên.
        </div>

        {/* Existing accounts list */}
        {cfg.accounts.length > 0 && (
          <div className="space-y-1.5">
            <p className="text-xs font-semibold uppercase tracking-wider text-emerald-700/80">
              Tài khoản hiện có ({cfg.accounts.length}) — #1 luôn được dùng trước
            </p>
            {cfg.accounts.map((a, i) => (
              <div key={`${a.profile}:${a.project_id}`} className="flex items-center gap-2 rounded-lg border border-emerald-200/60 bg-white/60 px-3 py-2">
                <span className={`shrink-0 inline-flex items-center justify-center min-w-[28px] h-5 px-1.5 rounded-md text-[11px] font-mono font-bold tabular-nums ${
                  i === 0 ? "bg-emerald-100 text-emerald-700 ring-1 ring-emerald-300" : "bg-slate-100 text-slate-500"
                }`}>
                  #{i + 1}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-stone-800">{a.label || a.profile}</div>
                  <div className="flex items-center gap-2 text-[11px] text-stone-500 font-mono">
                    <span>profile: {a.profile}</span>
                    <span>·</span>
                    <span className="truncate">project: {a.project_id}</span>
                  </div>
                </div>
                <Button
                  className="h-7 w-7 rounded-md bg-rose-50 text-rose-500 hover:bg-rose-100 p-0"
                  onClick={() => removeAccount(i)}
                  disabled={saving}
                >
                  <Trash2 className="size-3.5" />
                </Button>
              </div>
            ))}
          </div>
        )}

        {/* Reuse existing profile (Cách A) */}
        <div className="space-y-1 rounded-xl border border-emerald-200 bg-emerald-50/60 p-3">
          <p className="text-xs font-semibold text-emerald-800">Tái dùng profile đã onboard</p>
          <p className="text-[10px] text-emerald-700/70 leading-relaxed">
            Chọn profile Google đã có session (Flow/ChatGPT/Gemini) → tự lấy project_id + thêm vào pool, không cần đăng nhập.
          </p>
          <ReuseProfilePicker
            cs={{ url: cfg.captcha_solver_url, apiKey: cfg.captcha_solver_api_key }}
            onReuse={reuseAccount}
          />
        </div>

        {/* Add new account */}
        <div className="space-y-2 rounded-xl border border-dashed border-emerald-300 bg-white/40 p-3">
          <p className="text-xs font-semibold text-emerald-800">+ Thêm tài khoản mới</p>
          <div className="grid gap-2 sm:grid-cols-3">
            <div>
              <label className="text-[11px] text-stone-500">
                Label (chọn hoặc gõ)
                {!manuallyEditedLabel && cfg.accounts.length > 0 && (
                  <span className="ml-1 text-emerald-600">· gợi ý: {suggestedLabel}</span>
                )}
              </label>
              <Input
                value={draft.label || ""}
                onChange={(e) => {
                  setDraft({ ...draft, label: e.target.value });
                  setManuallyEditedLabel(true);
                }}
                placeholder={suggestedLabel}
                className="mt-1 h-8 rounded-lg border-stone-200 text-xs"
                list="flow-label-presets"
                autoComplete="off"
              />
              {/* Native HTML5 datalist — gõ thoải mái, dropdown gợi ý 6 preset
                  phổ biến + bất kỳ label nào đã dùng trước đó để khỏi đặt
                  trùng. */}
              {/* Preset labels phản ánh thứ tự fallback FIFO — Main luôn
                  #1, Backup là dự phòng đầu tiên, Spare N là các slot dự
                  bị tiếp theo trong rotation, Standby là account chờ. */}
              <datalist id="flow-label-presets">
                <option value="Main" />
                <option value="Backup" />
                <option value="Spare 1" />
                <option value="Spare 2" />
                <option value="Spare 3" />
                <option value="Standby" />
                {cfg.accounts
                  .map((a) => a.label || "")
                  .filter((v, i, arr) => v && arr.indexOf(v) === i)
                  .map((v) => (
                    <option key={`used-${v}`} value={v} />
                  ))}
              </datalist>
            </div>
            <div>
              <label className="text-[11px] text-stone-500">
                Profile (browser context)
                {!manuallyEditedProfile && cfg.accounts.length > 0 && (
                  <span className="ml-1 text-emerald-600">· gợi ý: {suggestedProfile}</span>
                )}
              </label>
              <Input
                value={draft.profile}
                onChange={(e) => {
                  setDraft({ ...draft, profile: e.target.value });
                  setManuallyEditedProfile(true);
                }}
                placeholder={suggestedProfile}
                className="mt-1 h-8 rounded-lg border-stone-200 text-xs font-mono"
              />
            </div>
            <div>
              <label className="text-[11px] text-stone-500">Project ID (Flow URL)</label>
              <Input
                value={draft.project_id}
                onChange={(e) => setDraft({ ...draft, project_id: e.target.value })}
                placeholder="54468d77-02ff-4a06-..."
                className="mt-1 h-8 rounded-lg border-stone-200 text-xs font-mono"
              />
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <Button
              className="h-8 rounded-lg bg-emerald-600 px-3 text-xs text-white hover:bg-emerald-700"
              onClick={addAccount}
              disabled={saving}
            >
              {saving ? <LoaderCircle className="size-3.5 animate-spin" /> : <Plus className="size-3.5" />}
              Thêm vào pool
            </Button>
            <Button
              className="h-8 rounded-lg border border-emerald-200 bg-white px-3 text-xs text-emerald-700 hover:bg-emerald-50"
              onClick={() => triggerManualLogin(false)}
            >
              <ExternalLink className="size-3.5" /> Mở noVNC + login thủ công
            </Button>
            <Button
              className="h-8 rounded-lg border border-amber-200 bg-white px-3 text-xs text-amber-700 hover:bg-amber-50"
              onClick={() => triggerManualLogin(true)}
              title="Kill Chrome cũ và mở lại — dùng khi noVNC hiển thị desktop trống (Connected... :99)"
            >
              <RotateCw className="size-3.5" /> Khởi động lại Chrome
            </Button>
          </div>
          <p className="text-[10px] text-stone-500 leading-relaxed">
            <b>Cách lấy project_id:</b> sau khi login Google trong noVNC, truy cập{" "}
            <code className="text-emerald-700">labs.google/fx/vi/tools/flow</code> → tạo project mới → copy UUID từ URL{" "}
            <code className="text-emerald-700">.../project/&lt;UUID&gt;</code>.
          </p>
        </div>

        {/* ── 1-CLICK ADD ACCOUNT ── primary onboarding path */}
        <div className="space-y-2 rounded-xl border-2 border-fuchsia-300 bg-gradient-to-br from-fuchsia-50/60 to-cyan-50/60 p-3">
          <div className="flex items-center justify-between">
            <p className="text-xs font-bold text-fuchsia-800 flex items-center gap-1.5">
              <Sparkles className="size-3.5" /> 1-click thêm tài khoản (tự động hoàn toàn)
            </p>
            <span className="text-[10px] text-fuchsia-700/80">
              auto-fill profile <code className="font-mono">{suggestedProfile}</code> · label <code className="font-mono">{suggestedLabel}</code>
            </span>
          </div>
          <p className="text-[10px] text-fuchsia-700/70 leading-relaxed">
            Login Google + tự lấy/tạo Flow project + tự add vào pool — chỉ cần email + mật khẩu.
            Khi gặp 2FA, dùng panel xanh chàm bên dưới để xử lý (số tap hoặc mã SMS).
          </p>
          <SavedAccountsSelect
            csUrl={cfg.captcha_solver_url}
            csApiKey={cfg.captcha_solver_api_key}
            selected={selectedAccount}
            onSelect={(email, acct) => {
              setSelectedAccount(email);
              setAutoLogin({ email: acct.email, password: acct.password, code: "", totpSecret: acct.totp_secret || "" });
            }}
            disabled={oneClickRunning}
            refreshKey={savedRefreshKey}
          />
          <div className="grid gap-2 sm:grid-cols-2">
            <div>
              <label className="text-[11px] text-stone-500">Email Google</label>
              <Input
                value={autoLogin.email}
                onChange={(e) => setAutoLogin({ ...autoLogin, email: e.target.value })}
                placeholder="you@gmail.com"
                className="mt-1 h-8 rounded-lg border-fuchsia-200 text-xs font-mono"
                autoComplete="off"
                disabled={oneClickRunning}
              />
            </div>
            <div>
              <label className="text-[11px] text-stone-500">Mật khẩu</label>
              <div className="relative">
                <Input
                  type={showPassword ? "text" : "password"}
                  value={autoLogin.password}
                  onChange={(e) => setAutoLogin({ ...autoLogin, password: e.target.value })}
                  placeholder="••••••••"
                  className="mt-1 h-8 rounded-lg border-fuchsia-200 text-xs font-mono pr-8"
                  autoComplete="off"
                  disabled={oneClickRunning}
                />
                <button
                  type="button"
                  className="absolute right-1.5 top-1/2 -translate-y-1/2 text-stone-400 hover:text-stone-600"
                  onClick={() => setShowPassword(!showPassword)}
                  tabIndex={-1}
                >
                  {showPassword ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />}
                </button>
              </div>
            </div>
          </div>
          <div>
            <label className="text-[11px] text-stone-500 flex items-center gap-1">
              <Shield className="size-3" /> TOTP Secret
            </label>
            <Input
              value={autoLogin.totpSecret}
              onChange={(e) => setAutoLogin({ ...autoLogin, totpSecret: e.target.value })}
              placeholder="xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx"
              className="mt-1 h-8 rounded-lg border-amber-200 text-xs font-mono bg-amber-50/30"
              autoComplete="off"
              disabled={oneClickRunning}
            />
            {totpCode && (
              <div className="mt-1 flex items-center gap-2">
                <span className="text-[11px] text-amber-700">Mã hiện tại:</span>
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-amber-100 text-amber-900 font-mono text-sm font-bold tracking-widest">
                  {totpCode}
                </span>
                <span className="text-[10px] text-amber-500">({totpRemaining}s)</span>
              </div>
            )}
          </div>
          <div className="flex gap-2">
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 rounded-lg text-[11px]"
              onClick={handleSaveAccount}
              disabled={isSavingAccount || !autoLogin.email.trim() || !autoLogin.password}
            >
              {isSavingAccount ? <LoaderCircle className="mr-1 size-3 animate-spin" /> : null}
              Lưu tài khoản
            </Button>
          </div>
          <Button
            className="w-full h-9 rounded-lg bg-gradient-to-r from-fuchsia-600 to-cyan-600 px-3 text-xs font-bold text-white hover:from-fuchsia-700 hover:to-cyan-700 shadow-lg shadow-fuchsia-200"
            onClick={oneClickAddAccount}
            disabled={oneClickRunning}
          >
            {oneClickRunning
              ? <><LoaderCircle className="size-3.5 animate-spin" /> Đang chạy…</>
              : <><Sparkles className="size-3.5" /> Tự động setup (1-click)</>}
          </Button>
          {oneClickStep && (
            <p className="text-[11px] text-fuchsia-800 bg-white/60 rounded-md px-2 py-1.5 font-mono">
              {oneClickStep}
            </p>
          )}
        </div>

        {/* ── Auto-login CLI (login only — for advanced users) ── */}
        <div className="space-y-2 rounded-xl border border-dashed border-indigo-300 bg-indigo-50/40 p-3">
          <div className="flex items-center justify-between">
            <p className="text-xs font-semibold text-indigo-800 flex items-center gap-1.5">
              <KeyRound className="size-3.5" /> Auto-login (CLI) — chỉ đăng nhập, không add pool
            </p>
            <span className="text-[10px] text-indigo-600/70">
              dùng profile <code className="font-mono">{draft.profile || "—"}</code>
            </span>
          </div>
          <p className="text-[10px] text-indigo-700/70 leading-relaxed">
            Backend Playwright tự điền email + mật khẩu, dừng lại khi gặp 2FA để bạn nhập mã hoặc bấm xác minh trên điện thoại.
            Nếu Google chặn (anti-bot), Chrome vẫn ở noVNC — bạn login thủ công nốt.
          </p>
          <SavedAccountsSelect
            csUrl={cfg.captcha_solver_url}
            csApiKey={cfg.captcha_solver_api_key}
            selected={selectedAccount}
            onSelect={(email, acct) => {
              setSelectedAccount(email);
              setAutoLogin({ email: acct.email, password: acct.password, code: "", totpSecret: acct.totp_secret || "" });
            }}
            disabled={loginSession?.state === "running" || loginSession?.state === "starting"}
            refreshKey={savedRefreshKey}
          />
          <div className="grid gap-2 sm:grid-cols-2">
            <div>
              <label className="text-[11px] text-stone-500">Email Google</label>
              <Input
                value={autoLogin.email}
                onChange={(e) => setAutoLogin({ ...autoLogin, email: e.target.value })}
                placeholder="you@gmail.com"
                className="mt-1 h-8 rounded-lg border-stone-200 text-xs font-mono"
                autoComplete="off"
              />
            </div>
            <div>
              <label className="text-[11px] text-stone-500">Mật khẩu</label>
              <div className="relative">
                <Input
                  type={showPassword ? "text" : "password"}
                  value={autoLogin.password}
                  onChange={(e) => setAutoLogin({ ...autoLogin, password: e.target.value })}
                  placeholder="••••••••"
                  className="mt-1 h-8 rounded-lg border-stone-200 text-xs font-mono pr-8"
                  autoComplete="off"
                />
                <button
                  type="button"
                  className="absolute right-1.5 top-1/2 -translate-y-1/2 text-stone-400 hover:text-stone-600"
                  onClick={() => setShowPassword(!showPassword)}
                  tabIndex={-1}
                >
                  {showPassword ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />}
                </button>
              </div>
            </div>
          </div>
          <div>
            <label className="text-[11px] text-stone-500 flex items-center gap-1">
              <Shield className="size-3" /> TOTP Secret
            </label>
            <Input
              value={autoLogin.totpSecret}
              onChange={(e) => setAutoLogin({ ...autoLogin, totpSecret: e.target.value })}
              placeholder="xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx"
              className="mt-1 h-8 rounded-lg border-amber-200 text-xs font-mono bg-amber-50/30"
              autoComplete="off"
              disabled={loginSession?.state === "running" || loginSession?.state === "starting"}
            />
            {totpCode && (
              <div className="mt-1 flex items-center gap-2">
                <span className="text-[11px] text-amber-700">Mã hiện tại:</span>
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-amber-100 text-amber-900 font-mono text-sm font-bold tracking-widest">
                  {totpCode}
                </span>
                <span className="text-[10px] text-amber-500">({totpRemaining}s)</span>
              </div>
            )}
          </div>
          <div className="flex gap-2 pt-1">
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 rounded-lg text-[11px]"
              onClick={handleSaveAccount}
              disabled={isSavingAccount || !autoLogin.email.trim() || !autoLogin.password}
            >
              {isSavingAccount ? <LoaderCircle className="mr-1 size-3 animate-spin" /> : null}
              Lưu tài khoản
            </Button>
          </div>
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <Button
              className="h-8 rounded-lg bg-indigo-600 px-3 text-xs text-white hover:bg-indigo-700"
              onClick={startAutoLogin}
              disabled={loginSession?.state === "running" || loginSession?.state === "starting"}
            >
              {loginSession?.state === "running" || loginSession?.state === "starting"
                ? <LoaderCircle className="size-3.5 animate-spin" />
                : <KeyRound className="size-3.5" />}
              Bắt đầu auto-login
            </Button>
            {loginSession && loginSession.state !== "none" && (
              <Button
                className="h-8 rounded-lg border border-stone-200 bg-white px-3 text-xs text-stone-600 hover:bg-stone-50"
                onClick={cancelLoginSession}
              >
                <X className="size-3.5" /> Đóng phiên
              </Button>
            )}
          </div>

          {/* Status panel — chỉ hiện khi có phiên */}
          {loginSession && loginSession.state !== "none" && (
            <div className={`mt-2 rounded-lg border p-3 text-xs space-y-2 ${
              loginSession.state === "success" ? "border-emerald-300 bg-emerald-50/70"
              : loginSession.state === "failed" ? "border-rose-300 bg-rose-50/70"
              : loginSession.state === "need_tap" ? "border-violet-300 bg-violet-50/70"
              : loginSession.state === "need_code" ? "border-amber-300 bg-amber-50/70"
              : "border-indigo-200 bg-white/80"
            }`}>
              <div className="flex items-center gap-2">
                <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider ${
                  loginSession.state === "success" ? "bg-emerald-100 text-emerald-700"
                  : loginSession.state === "failed" ? "bg-rose-100 text-rose-700"
                  : loginSession.state === "need_tap" ? "bg-violet-100 text-violet-700"
                  : loginSession.state === "need_code" ? "bg-amber-100 text-amber-700"
                  : "bg-indigo-100 text-indigo-700"
                }`}>
                  {(loginSession.state === "running" || loginSession.state === "starting") && (
                    <LoaderCircle className="size-3 animate-spin" />
                  )}
                  {loginSession.state}
                </span>
                <span className="text-stone-600">{loginSession.message}</span>
                {typeof loginSession.elapsed_sec === "number" && (
                  <span className="ml-auto text-[10px] text-stone-400 font-mono">{loginSession.elapsed_sec}s</span>
                )}
              </div>

              {loginSession.state === "need_tap" && (
                <div className="flex items-center gap-2 rounded-md bg-violet-100/60 px-2 py-1.5">
                  <Smartphone className="size-4 text-violet-700" />
                  <span className="text-violet-900">
                    Mở app Gmail/Google trên điện thoại
                    {loginSession.tap_number ? (
                      <> và bấm số <b className="text-base font-mono">{loginSession.tap_number}</b></>
                    ) : (
                      <> và bấm "Có" để xác minh</>
                    )}
                  </span>
                </div>
              )}

              {loginSession.state === "need_code" && (
                <div className="flex items-end gap-2">
                  <div className="flex-1">
                    <label className="text-[11px] text-amber-800">Mã 2FA (SMS hoặc Authenticator)</label>
                    <Input
                      value={autoLogin.code}
                      onChange={(e) => setAutoLogin({ ...autoLogin, code: e.target.value })}
                      placeholder="123456"
                      className="mt-1 h-8 rounded-lg border-amber-200 text-xs font-mono"
                      autoComplete="off"
                      onKeyDown={(e) => { if (e.key === "Enter") void submit2faCode(); }}
                    />
                  </div>
                  <Button
                    className="h-8 rounded-lg bg-amber-600 px-3 text-xs text-white hover:bg-amber-700"
                    onClick={submit2faCode}
                  >
                    Gửi mã
                  </Button>
                </div>
              )}

              {loginSession.state === "failed" && loginSession.error && (
                <p className="text-rose-700 text-[11px]">{loginSession.error}</p>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
