"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { LoaderCircle, KeyRound, Sparkles, Smartphone, X, ExternalLink, Shield, Eye, EyeOff, RefreshCw, Timer } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { request } from "@/lib/request";
import { SavedAccountsSelect } from "@/components/saved-accounts-select";
import { generateTotpCode, totpSecondsRemaining } from "@/lib/totp";

type OnboardState = {
  profile: string;
  email: string;
  state: "none" | "starting" | "running" | "need_tap" | "need_code" | "success" | "failed";
  message: string;
  tap_number?: string | null;
  elapsed_sec?: number;
  error?: string | null;
  access_token?: string | null;
  expires?: string | null;
  captured_email?: string | null;
  access_token_preview?: string | null;
};

type CaptchaSolverCfg = {
  url: string;
  apiKey: string;
};

export function ChatGPTOnboardCard() {
  // Reuse the same captcha-solver creds the Flow card stores under
  // providers.flow — admins shouldn't have to enter them twice.
  const [cs, setCs] = useState<CaptchaSolverCfg>({
    url: "http://172.16.10.38:8010",
    apiKey: "",
  });
  const [draft, setDraft] = useState({ email: "", password: "", code: "", totpSecret: "" });
  const [running, setRunning] = useState(false);
  const [session, setSession] = useState<OnboardState | null>(null);
  const pollRef = useRef<number | null>(null);
  const [totpCode, setTotpCode] = useState("");
  const [totpRemaining, setTotpRemaining] = useState(30);
  const totpTimerRef = useRef<number | null>(null);
  const [selectedAccount, setSelectedAccount] = useState("");
  const [showPassword, setShowPassword] = useState(true);
  const [isSavingAccount, setIsSavingAccount] = useState(false);
  const [savedRefreshKey, setSavedRefreshKey] = useState(0);

  // ── Auto-refresh state ──
  const [autoRefreshRunning, setAutoRefreshRunning] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(30);

  async function checkAutoRefreshStatus() {
    try {
      const res = await fetch(`${cs.url}/v1/chatgpt/auto-refresh/status`, {
        headers: { Authorization: `Bearer ${cs.apiKey}` },
      });
      if (res.ok) {
        const data = await res.json();
        setAutoRefreshRunning(data.running);
      }
    } catch { /* ignore */ }
  }

  async function toggleAutoRefresh() {
    try {
      if (autoRefreshRunning) {
        await fetch(`${cs.url}/v1/chatgpt/auto-refresh/stop`, {
          method: "POST",
          headers: { Authorization: `Bearer ${cs.apiKey}` },
        });
        setAutoRefreshRunning(false);
        toast.success("Đã dừng auto-refresh");
      } else {
        await fetch(`${cs.url}/v1/chatgpt/auto-refresh/start?interval_minutes=${refreshInterval}`, {
          method: "POST",
          headers: { Authorization: `Bearer ${cs.apiKey}` },
        });
        setAutoRefreshRunning(true);
        toast.success(`Auto-refresh mỗi ${refreshInterval} phút`);
      }
    } catch { toast.error("Lỗi auto-refresh"); }
  }

  async function refreshNow() {
    if (!draft.email.trim()) {
      toast.error("Chọn tài khoản trước");
      return;
    }
    const profile = profileSuggestion();
    toast.info(`Đang refresh token cho ${profile}...`);
    try {
      const res = await fetch(`${cs.url}/v1/chatgpt/${encodeURIComponent(profile)}/refresh-jwt`, {
        headers: { Authorization: `Bearer ${cs.apiKey}` },
      });
      const data = await res.json();
      if (data.ok) {
        toast.success(`Refresh OK (${data.method}): ${data.access_token_preview}`);
      } else {
        toast.error(`Refresh fail: ${data.error}`);
      }
    } catch { toast.error("Lỗi gọi refresh"); }
  }

  // Check auto-refresh status on mount
  useEffect(() => {
    void checkAutoRefreshStatus();
  }, []);

  // Auto-refresh TOTP code when secret is provided
  const refreshTotp = useCallback(async (secret: string) => {
    if (!secret.trim()) { setTotpCode(""); return; }
    try {
      const code = await generateTotpCode(secret);
      setTotpCode(code);
      setTotpRemaining(totpSecondsRemaining());
    } catch { setTotpCode(""); }
  }, []);

  // Keep TOTP code live
  useEffect(() => {
    if (!draft.totpSecret.trim()) { setTotpCode(""); return; }
    void refreshTotp(draft.totpSecret);
    totpTimerRef.current = window.setInterval(() => {
      void refreshTotp(draft.totpSecret);
    }, 5000);
    return () => {
      if (totpTimerRef.current) window.clearInterval(totpTimerRef.current);
    };
  }, [draft.totpSecret, refreshTotp]);

  useEffect(() => {
    void fetchCfg();
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  async function fetchCfg() {
    try {
      const data = await request.get("/api/settings");
      const flow = ((data.data as any)?.config?.providers || {}).flow || {};
      setCs({
        url: flow.captcha_solver_url || "http://172.16.10.38:8010",
        apiKey: flow.captcha_solver_api_key || "",
      });
    } catch (e) {
      console.error(e);
    }
  }

  function stopPolling() {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  function profileSuggestion() {
    // chatgpt-<localpart-of-email>
    const local = (draft.email.split("@")[0] || "default").replace(/[^a-z0-9-]/gi, "-");
    return `chatgpt-${local}`;
  }

  async function pollOnboardStatus(profile: string, onSuccess: (s: OnboardState) => void) {
    try {
      const res = await fetch(`${cs.url}/v1/chatgpt/${encodeURIComponent(profile)}/onboard-status`, {
        headers: { Authorization: `Bearer ${cs.apiKey}` },
      });
      if (!res.ok) return;
      const data: OnboardState = await res.json();
      setSession(data);
      if (data.state === "success" || data.state === "failed") {
        stopPolling();
        if (data.state === "success") {
          toast.success(`Login ChatGPT OK (${data.captured_email})`);
          onSuccess(data);
        } else {
          toast.error(`Login fail: ${data.error || data.message}`);
          setRunning(false);
        }
      }
    } catch {
      /* network blip — keep polling */
    }
  }

  async function onboardAndAddToPool() {
    if (!draft.email.trim() || !draft.password) {
      toast.error("Cần email + mật khẩu Google");
      return;
    }
    const profile = profileSuggestion();
    stopPolling();
    setRunning(true);
    setSession(null);
    try {
      // 1) Start captcha-solver onboard
      const res = await fetch(`${cs.url}/v1/chatgpt/onboard`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${cs.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          profile,
          email: draft.email.trim(),
          password: draft.password,
          totp_secret: draft.totpSecret.trim(),
        }),
      });
      if (!res.ok) throw new Error(`onboard HTTP ${res.status}`);
      const initial = await res.json();
      setSession(initial);
      // Open noVNC so user can monitor / handle anti-bot challenges
      const noVncUrl = cs.url.replace(":8010", ":6080") + "/vnc.html?autoconnect=1";
      window.open(noVncUrl, "_blank", "noopener,width=1024,height=720");

      // 2) Poll for success
      const handleSuccess = async (s: OnboardState) => {
        if (!s.access_token) {
          toast.error("Login OK nhưng không có access_token");
          setRunning(false);
          return;
        }
        // 3) POST token to chatgpt2api accounts pool
        try {
          await request.post("/api/accounts", { tokens: [s.access_token] });
          toast.success(`Đã thêm account ${s.captured_email} vào pool 🎉`);
          setDraft({ email: "", password: "", code: "", totpSecret: "" });
        } catch (e: any) {
          toast.error(`Add to pool fail: ${e?.message || e}`);
        } finally {
          setRunning(false);
        }
      };
      pollRef.current = window.setInterval(() => {
        void pollOnboardStatus(profile, handleSuccess);
      }, 1500);
    } catch (e: any) {
      toast.error(`Onboard error: ${e?.message}`);
      setRunning(false);
    }
  }

  async function submit2faCode() {
    if (!session?.profile || !draft.code.trim()) {
      toast.error("Cần mã 2FA");
      return;
    }
    try {
      const res = await fetch(
        `${cs.url}/v1/chatgpt/${encodeURIComponent(session.profile)}/onboard-2fa-code`,
        {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${cs.apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ code: draft.code.trim() }),
        },
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      toast.success("Đã gửi mã, đợi xác minh…");
      setDraft({ ...draft, code: "" });
    } catch (e: any) {
      toast.error(`Lỗi gửi mã: ${e?.message}`);
    }
  }

  function cancelSession() {
    stopPolling();
    setSession(null);
    setRunning(false);
    setDraft({ email: "", password: "", code: "", totpSecret: "" });
  }

  async function handleSaveAccount() {
    if (!draft.email.trim() || !draft.password) {
      toast.error("Cần email + mật khẩu để lưu");
      return;
    }
    setIsSavingAccount(true);
    try {
      await fetch(`${cs.url}/v1/accounts/saved`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${cs.apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({ email: draft.email.trim(), password: draft.password, totp_secret: draft.totpSecret.trim() }),
      });
      toast.success("Đã lưu tài khoản");
      setDraft({ email: "", password: "", code: "", totpSecret: "" });
      setSelectedAccount("");
      setSavedRefreshKey((k) => k + 1);
    } catch {
      toast.error("Lưu tài khoản thất bại");
    } finally {
      setIsSavingAccount(false);
    }
  }

  function openNoVNC() {
    const noVncUrl = cs.url.replace(":8010", ":6080") + "/vnc.html?autoconnect=1";
    window.open(noVncUrl, "_blank");
  }

  return (
    <Card className="rounded-3xl border-blue-100/80 bg-blue-50/30">
      <CardContent className="space-y-4 p-5">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="size-4 text-blue-600" />
              <h3 className="text-sm font-semibold text-blue-900">ChatGPT via Google OAuth</h3>
            </div>
            <p className="text-xs text-blue-700/70 mt-0.5">
              Tự động login chat.openai.com bằng tài khoản Google → scrape JWT access_token → add vào pool ChatGPT free.
              Bypass hoàn toàn 24KB session-token limit.
            </p>
          </div>
        </div>

        {/* 1-click form */}
        <div className="space-y-2 rounded-xl border-2 border-blue-300 bg-gradient-to-br from-blue-50/60 to-cyan-50/60 p-3">
          <p className="text-xs font-bold text-blue-800 flex items-center gap-1.5">
            <KeyRound className="size-3.5" /> 1-click thêm ChatGPT free (qua Google)
          </p>
          <p className="text-[10px] text-blue-700/70 leading-relaxed">
            Nhập email + mật khẩu Google. Backend Playwright tự click "Continue with Google" trên chat.openai.com,
            login qua trang Google (cùng flow như Flow), redirect về chatgpt.com, scrape JWT, save vào pool.
            Khi gặp 2FA, dùng panel xanh chàm bên dưới.
          </p>

          {/* Saved accounts dropdown */}
          <SavedAccountsSelect
            csUrl={cs.url}
            csApiKey={cs.apiKey}
            selected={selectedAccount}
            onSelect={(email, acct) => {
              setSelectedAccount(email);
              setDraft({ email: acct.email, password: acct.password, code: "", totpSecret: acct.totp_secret || "" });
            }}
            disabled={running}
            refreshKey={savedRefreshKey}
          />

          <div className="grid gap-2 sm:grid-cols-2">
            <div>
              <label className="text-[11px] text-stone-500">Email Google</label>
              <Input
                value={draft.email}
                onChange={(e) => setDraft({ ...draft, email: e.target.value })}
                placeholder="you@gmail.com"
                className="mt-1 h-8 rounded-lg border-blue-200 text-xs font-mono"
                autoComplete="off"
                disabled={running}
              />
            </div>
            <div>
              <label className="text-[11px] text-stone-500">Mật khẩu Google</label>
              <div className="relative">
                <Input
                  type={showPassword ? "text" : "password"}
                  value={draft.password}
                  onChange={(e) => setDraft({ ...draft, password: e.target.value })}
                  placeholder="••••••••"
                  className="mt-1 h-8 rounded-lg border-blue-200 text-xs font-mono pr-8"
                  autoComplete="off"
                  disabled={running}
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
              <Shield className="size-3" /> TOTP Secret (Authenticator — bỏ qua dấu cách, tự sinh mã khi bật)
            </label>
            <Input
              value={draft.totpSecret}
              onChange={(e) => setDraft({ ...draft, totpSecret: e.target.value })}
              placeholder="xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx"
              className="mt-1 h-8 rounded-lg border-amber-200 text-xs font-mono bg-amber-50/30"
              autoComplete="off"
              disabled={running}
            />
            {totpCode && (
              <div className="mt-1 flex items-center gap-2">
                <span className="text-[11px] text-amber-700">Ma hien tai:</span>
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-amber-100 text-amber-900 font-mono text-sm font-bold tracking-widest">
                  {totpCode}
                </span>
                <span className="text-[10px] text-amber-500">({totpRemaining}s)</span>
              </div>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 rounded-lg text-[11px]"
              onClick={handleSaveAccount}
              disabled={isSavingAccount || !draft.email.trim() || !draft.password}
            >
              {isSavingAccount ? <LoaderCircle className="mr-1 size-3 animate-spin" /> : null}
              Lưu tài khoản
            </Button>
            <Button
              className="h-9 rounded-lg bg-gradient-to-r from-blue-600 to-cyan-600 px-3 text-xs font-bold text-white hover:from-blue-700 hover:to-cyan-700 shadow-lg shadow-blue-200"
              onClick={onboardAndAddToPool}
              disabled={running}
            >
              {running
                ? <><LoaderCircle className="size-3.5 animate-spin" /> Đang chạy…</>
                : <><Sparkles className="size-3.5" /> Tự động setup (1-click)</>}
            </Button>
            <Button
              className="h-9 rounded-lg border border-blue-200 bg-white px-3 text-xs text-blue-700 hover:bg-blue-50"
              onClick={openNoVNC}
            >
              <ExternalLink className="size-3.5" /> Mở noVNC
            </Button>
            <Button
              className="h-9 rounded-lg border border-purple-200 bg-white px-3 text-xs text-purple-700 hover:bg-purple-50"
              onClick={refreshNow}
              disabled={running || !draft.email.trim()}
            >
              <RefreshCw className="size-3.5" /> Refresh ngay
            </Button>
            {session && session.state !== "none" && (
              <Button
                className="h-9 rounded-lg border border-stone-200 bg-white px-3 text-xs text-stone-600 hover:bg-stone-50"
                onClick={cancelSession}
              >
                <X className="size-3.5" /> Đóng phiên
              </Button>
            )}
          </div>

          {/* Auto-refresh controls */}
          <div className="flex items-center gap-2 pt-1 border-t border-blue-200/50">
            <Timer className="size-3.5 text-purple-600" />
            <span className="text-[11px] text-stone-600">Auto-refresh token:</span>
            <button
              onClick={toggleAutoRefresh}
              className={`px-2 py-0.5 rounded text-[10px] font-semibold transition-colors ${
                autoRefreshRunning
                  ? "bg-emerald-100 text-emerald-700 hover:bg-emerald-200"
                  : "bg-stone-100 text-stone-500 hover:bg-stone-200"
              }`}
            >
              {autoRefreshRunning ? "ON" : "OFF"}
            </button>
            {!autoRefreshRunning && (
              <>
                <span className="text-[10px] text-stone-400">mỗi</span>
                <input
                  type="number"
                  value={refreshInterval}
                  onChange={(e) => setRefreshInterval(Math.max(5, Math.min(120, Number(e.target.value))))}
                  className="w-12 h-6 rounded border border-stone-200 text-center text-[10px]"
                  min={5}
                  max={120}
                />
                <span className="text-[10px] text-stone-400">phút</span>
              </>
            )}
          </div>

          {/* Status panel */}
          {session && session.state !== "none" && (
            <div className={`mt-2 rounded-lg border p-3 text-xs space-y-2 ${
              session.state === "success" ? "border-emerald-300 bg-emerald-50/70"
              : session.state === "failed" ? "border-rose-300 bg-rose-50/70"
              : session.state === "need_tap" ? "border-violet-300 bg-violet-50/70"
              : session.state === "need_code" ? "border-amber-300 bg-amber-50/70"
              : "border-blue-200 bg-white/80"
            }`}>
              <div className="flex items-center gap-2">
                <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider ${
                  session.state === "success" ? "bg-emerald-100 text-emerald-700"
                  : session.state === "failed" ? "bg-rose-100 text-rose-700"
                  : session.state === "need_tap" ? "bg-violet-100 text-violet-700"
                  : session.state === "need_code" ? "bg-amber-100 text-amber-700"
                  : "bg-blue-100 text-blue-700"
                }`}>
                  {(session.state === "running" || session.state === "starting") && (
                    <LoaderCircle className="size-3 animate-spin" />
                  )}
                  {session.state}
                </span>
                <span className="text-stone-600">{session.message}</span>
                {typeof session.elapsed_sec === "number" && (
                  <span className="ml-auto text-[10px] text-stone-400 font-mono">{session.elapsed_sec}s</span>
                )}
              </div>

              {session.state === "need_tap" && (
                <div className="flex items-center gap-2 rounded-md bg-violet-100/60 px-2 py-1.5">
                  <Smartphone className="size-4 text-violet-700" />
                  <span className="text-violet-900">
                    Mở app Gmail/Google trên điện thoại
                    {session.tap_number ? (
                      <> và bấm số <b className="text-base font-mono">{session.tap_number}</b></>
                    ) : (
                      <> và bấm "Có" để xác minh</>
                    )}
                  </span>
                </div>
              )}

              {session.state === "need_code" && (
                <div className="space-y-2">
                  {/* Auto-generated TOTP code when secret is available */}
                  {totpCode && (
                    <div className="flex items-center gap-2 rounded-md bg-amber-100/70 px-2 py-1.5">
                      <Shield className="size-4 text-amber-700" />
                      <span className="text-amber-800 text-[11px]">Ma tu sinh tu TOTP secret:</span>
                      <span className="text-amber-900 font-mono text-lg font-bold tracking-widest">
                        {totpCode}
                      </span>
                      <span className="text-[10px] text-amber-500 ml-auto">({totpRemaining}s)</span>
                    </div>
                  )}
                  {/* Manual fallback input */}
                  <div className="flex items-end gap-2">
                    <div className="flex-1">
                      <label className="text-[11px] text-amber-800">
                        {totpCode ? "Hoac nhap thu cong:" : "Ma 2FA (SMS hoac Authenticator)"}
                      </label>
                      <Input
                        value={draft.code}
                        onChange={(e) => setDraft({ ...draft, code: e.target.value })}
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
                </div>
              )}

              {session.state === "success" && session.access_token_preview && (
                <div className="text-emerald-700 text-[11px] font-mono break-all">
                  ✓ Token captured: {session.access_token_preview}
                  <div className="text-[10px] text-emerald-600 mt-0.5">Đã add vào pool. Email: {session.captured_email}</div>
                </div>
              )}

              {session.state === "failed" && session.error && (
                <p className="text-rose-700 text-[11px]">{session.error}</p>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
