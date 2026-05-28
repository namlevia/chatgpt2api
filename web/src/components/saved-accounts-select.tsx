"use client";

import { useEffect, useState } from "react";
import { Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";

type SavedAccount = {
  id: number;
  email: string;
  totp_secret: string;
  label: string;
};

type Props = {
  csUrl: string;
  csApiKey: string;
  selected: string;
  onSelect: (email: string, account: { email: string; password: string; totp_secret: string }) => void;
  disabled?: boolean;
  refreshKey?: number;
};

const STORAGE_KEY = "chatgpt2api_saved_accounts_cache";

function cacheAccounts(accounts: SavedAccount[]) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(accounts)); } catch {}
}

function loadCached(): SavedAccount[] {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); } catch { return []; }
}

export function SavedAccountsSelect({ csUrl, csApiKey, selected, onSelect, disabled, refreshKey }: Props) {
  const [accounts, setAccounts] = useState<SavedAccount[]>(loadCached());

  useEffect(() => {
    void fetchAccounts();
  }, [csUrl, refreshKey]);

  async function fetchAccounts() {
    try {
      const res = await fetch(`${csUrl}/v1/accounts/saved`, {
        headers: { Authorization: `Bearer ${csApiKey}` },
      });
      if (res.ok) {
        const data = await res.json();
        setAccounts(data);
        cacheAccounts(data);
      }
    } catch { /* ignore */ }
  }

  async function loadAccount(email: string) {
    if (!email) {
      onSelect("", { email: "", password: "", totp_secret: "" });
      return;
    }
    try {
      const res = await fetch(`${csUrl}/v1/accounts/saved/${encodeURIComponent(email)}`, {
        headers: { Authorization: `Bearer ${csApiKey}` },
      });
      if (res.ok) {
        const acct = await res.json();
        onSelect(email, acct);
      }
    } catch { toast.error("Không load được tài khoản"); }
  }

  async function deleteAccount(email: string) {
    try {
      await fetch(`${csUrl}/v1/accounts/saved/${encodeURIComponent(email)}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${csApiKey}` },
      });
      toast.success("Đã xóa");
      if (selected === email) {
        onSelect("", { email: "", password: "", totp_secret: "" });
      }
      fetchAccounts();
    } catch { toast.error("Lỗi xóa"); }
  }

  return (
    <div className="flex items-end gap-1.5">
      <div className="flex-1">
        <label className="text-[11px] text-stone-500">Tai khoan da luu</label>
        <select
          value={selected}
          onChange={(e) => loadAccount(e.target.value)}
          className="mt-1 h-8 w-full rounded-lg border border-stone-200 bg-white text-xs font-mono px-2 text-stone-700"
          disabled={disabled}
        >
          <option value="">-- Chon tai khoan ({accounts.length}) --</option>
          {accounts.map((a) => (
            <option key={a.id} value={a.email}>{a.label || a.email}</option>
          ))}
        </select>
      </div>
      {selected && (
        <Button
          size="sm"
          variant="ghost"
          className="h-8 px-2 text-[10px] text-rose-500 hover:bg-rose-50"
          onClick={() => deleteAccount(selected)}
        >
          <Trash2 className="size-3" />
        </Button>
      )}
    </div>
  );
}
