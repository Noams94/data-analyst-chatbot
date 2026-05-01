"use client";

/**
 * Chat view, scoped to a specific chat id from the URL. Loads the chat +
 * dataset on mount, then streams new messages via useChatStream.
 */

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Database, FileText, Loader2 } from "lucide-react";
import { PromptInput } from "@/components/prompt-input";
import { ChatMessage } from "@/components/chat-message";
import { ReportDialog } from "@/components/report-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useChatStream } from "@/lib/use-chat-stream";
import { useApi, type DatasetSummary } from "@/lib/api";

export default function ChatPage() {
  const params = useParams<{ id: string }>();
  const chatId = params.id;
  const router = useRouter();
  const api = useApi();
  const { state, send, stop, reset, setPinned } = useChatStream();
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [reportOpen, setReportOpen] = useState(false);
  const scrollerRef = useRef<HTMLDivElement>(null);

  // Load chat + dataset on mount; reload when the chat id changes.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    (async () => {
      try {
        const chat = await api.getChat(chatId);
        if (cancelled) return;
        const ds = await api.getDataset(chat.datasetId);
        if (cancelled) return;
        setDataset(ds);
        reset(chat.messages);
      } catch {
        if (!cancelled) router.replace("/app");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [api, chatId, reset, router]);

  // Scroll to bottom on new content.
  useEffect(() => {
    scrollerRef.current?.scrollTo({
      top: scrollerRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [state.messages.length, state.streaming?.content, state.streaming?.charts.length]);

  if (loading || !dataset) {
    return (
      <main className="flex flex-1 items-center justify-center text-muted-foreground">
        <Loader2 className="h-5 w-5 animate-spin" />
      </main>
    );
  }

  const busy = state.streaming?.isStreaming ?? false;

  return (
    <main className="flex flex-1 flex-col min-w-0">
      <div className="flex items-center justify-between gap-4 border-b px-6 py-2 text-sm">
        <div className="flex items-center gap-2 text-muted-foreground min-w-0">
          <Database className="h-4 w-4 shrink-0" />
          <span className="font-medium text-foreground truncate">{dataset.name}</span>
          <Badge variant="secondary" className="shrink-0">
            {dataset.rowCount.toLocaleString()} rows · {dataset.columns.length} cols
          </Badge>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          {state.messages.length > 0 ? (
            <Button size="sm" variant="outline" onClick={() => setReportOpen(true)}>
              <FileText className="h-4 w-4" /> Generate report
            </Button>
          ) : null}
        </div>
      </div>

      <ReportDialog
        chatId={chatId}
        open={reportOpen}
        onOpenChange={setReportOpen}
        defaultTitle={`${dataset.name} — Analysis report`}
      />

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
            placeholder="Ask about your data…"
          />
        </div>
      </div>
    </main>
  );
}
