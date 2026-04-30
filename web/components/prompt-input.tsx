"use client";

import { useState, type FormEvent, type KeyboardEvent } from "react";
import { Send, Square } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";

export function PromptInput({
  onSubmit,
  onStop,
  busy,
  placeholder = "Ask about your data…",
}: {
  onSubmit: (text: string) => void;
  onStop?: () => void;
  busy: boolean;
  placeholder?: string;
}) {
  const [text, setText] = useState("");

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    onSubmit(trimmed);
    setText("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <form onSubmit={submit} className="flex gap-2 items-end">
      <Textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        rows={1}
        className="resize-none min-h-[44px] max-h-40"
      />
      {busy && onStop ? (
        <Button type="button" variant="outline" onClick={onStop} aria-label="Stop">
          <Square className="h-4 w-4" />
        </Button>
      ) : (
        <Button type="submit" disabled={!text.trim() || busy} aria-label="Send">
          <Send className="h-4 w-4" />
        </Button>
      )}
    </form>
  );
}
