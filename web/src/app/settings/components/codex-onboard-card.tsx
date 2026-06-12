"use client";

import { Save } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { useSettingsStore } from "../store";

export function CodexOnboardCard() {
  const config = useSettingsStore((state) => state.config);
  const setField = useSettingsStore((state) => state.setField);

  return (
    <Card className="rounded-3xl border-slate-100/80 bg-slate-50/30">
      <CardContent className="space-y-4 p-5">
        <div className="space-y-2">
          <label className="text-sm font-semibold text-slate-800">Danh sách Tài khoản Codex (Tự động Đăng nhập Hàng loạt)</label>
          <Textarea
            value={String(config?.codex_auto_list || "")}
            onChange={(event) => setField("codex_auto_list", event.target.value)}
            placeholder="Ví dụ:&#10;acc1@outlook.com|pass1|receiver1@gmail.com|apppass1&#10;acc2@outlook.com|pass2|receiver2@gmail.com|apppass2"
            className="min-h-[200px] rounded-xl border-stone-200 bg-white font-mono text-xs"
          />
          <p className="text-xs text-stone-500 leading-relaxed mt-2">
            Mỗi dòng 1 tài khoản theo định dạng: <code className="bg-stone-100 px-1 py-0.5 rounded text-stone-700">codex_email|codex_pass|imap_email|imap_app_pass</code>.<br/>
            Tính năng Auto-Login ở tab <b>Tài khoản</b> sẽ tự động đồng bộ danh sách này để chạy. Nếu để trống IMAP, hệ thống sẽ dùng IMAP chung mà bạn cấu hình ở tab Tài khoản. Đừng quên bấm <b>Lưu cài đặt</b> (ở nút góc trên/dưới) sau khi chỉnh sửa nhé!
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
