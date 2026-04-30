"use client";

import { useEffect, useRef, useState } from "react";
import { Database, FileText } from "lucide-react";
import { UploadDropzone } from "@/components/upload-dropzone";
import { PromptInput } from "@/components/prompt-input";
import { ChatMessage } from "@/components/chat-message";
import { ReportDialog } from "@/components/report-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useChatStream } from "@/lib/use-chat-stream";
import { useApi, type DatasetSummary } from "@/lib/api";

export default function AppHome() {
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [chatId, setChatId] = useState<string | null>(null);
  const [reportOpen, setReportOpen] = useState(false);
  const api = useApi();
  const { state, send, stop, reset, setPinned } = useChatStream();
  const scrollerRef = useRef<HTMLDivElement>(null);

  // On first mount, hydrate from ?chatId= so a refreshed page restores its history.
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("chatId");
    if (!id) return;
    let cancelled = false;
    (async () => {
      try {
        const chat = await api.getChat(id);
        if (cancelled) return;
        const ds = await api.getDataset(chat.datasetId);
        if (cancelled) return;
        setDataset(ds);
        setChatId(chat.id);
        reset(chat.messages);
      } catch {
        // Stale URL; ignore and let the dropzone render.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [api, reset]);

  // After upload, immediately create a chat so the user can start typing,
  // and reflect the chatId in the URL for refresh-survival.
  useEffect(() => {
    if (!dataset || chatId) return;
    let cancelled = false;
    api.createChat(dataset.id).then((chat) => {
      if (cancelled) return;
      setChatId(chat.id);
      const url = new URL(window.location.href);
      url.searchParams.set("chatId", chat.id);
      window.history.replaceState({}, "", url.toString());
    });
    return () => {
      cancelled = true;
    };
  }, [api, dataset, chatId]);

  // Scroll to bottom on new content.
  useEffect(() => {
    scrollerRef.current?.scrollTo({
      top: scrollerRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [state.messages.length, state.streaming?.content, state.streaming?.charts.length]);

  if (!dataset) {
    return (
      <main className="flex flex-1 items-center justify-center px-6">
        <UploadDropzone onUploaded={setDataset} />
      </main>
    );
  }

  const busy = state.streaming?.isStreaming ?? false;

  return (
    <main className="flex flex-1 flex-col">
      <div className="flex items-center justify-between gap-4 border-b px-6 py-2 text-sm">
        <div className="flex items-center gap-2 text-muted-foreground min-w-0">
          <Database className="h-4 w-4 shrink-0" />
          <span className="font-medium text-foreground truncate">{dataset.name}</span>
          <Badge variant="secondary" className="shrink-0">
            {dataset.rowCount.toLocaleString()} rows · {dataset.columns.length} cols
          </Badge>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          {chatId && state.messages.length > 0 ? (
            <Button
              size="sm"
              variant="outline"
              onClick={() => setReportOpen(true)}
            >
              <FileText className="h-4 w-4" /> Generate report
            </Button>
          ) : null}
          <button
            className="text-muted-foreground hover:text-foreground whitespace-nowrap"
            onClick={() => {
              setDataset(null);
              setChatId(null);
              reset([]);
              const url = new URL(window.location.href);
              url.searchParams.delete("chatId");
              window.history.replaceState({}, "", url.toString());
            }}
          >
            Change dataset
          </button>
        </div>
      </div>
      {chatId ? (
        <ReportDialog
          chatId={chatId}
          open={reportOpen}
          onOpenChange={setReportOpen}
          defaultTitle={`${dataset.name} — Analysis report`}
        />
      ) : null}

      <div ref={scrollerRef} className="flex-1 overflow-y-auto px-6 py-4">
        <div className="max-w-3xl mx-auto">
          {state.messages.length === 0 && !state.streaming ? (
            <div className="text-center text-muted-foreground py-12">
              <p>Ask anything about your data — start with &ldquo;What columns are in this dataset?&rdquo;</p>
            </div>
          ) : null}
          {state.messages.map((m) => (
            <ChatMessage key={m.id} message={m} onPinToggle={setPinned} />
          ))}
          {state.streaming ? <ChatMessage message={state.streaming} /> : null}
        </div>
      </div>

      <div className="border-t px-6 py-3">
        <div className="max-w-3xl mx-auto">
          <PromptInput
            onSubmit={(text) => {
              if (chatId) void send(chatId, text);
            }}
            onStop={stop}
            busy={busy}
            placeholder={chatId ? "Ask about your data…" : "Preparing chat…"}
          />
        </div>
      </div>
    </main>
  );
}
