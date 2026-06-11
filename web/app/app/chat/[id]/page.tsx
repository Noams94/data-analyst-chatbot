"use client";

/**
 * Chat view, scoped to a specific chat id from the URL. Loads the chat +
 * dataset on mount, then streams new messages via useChatStream.
 */

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Database, Loader2, PanelRightClose, PanelRightOpen } from "lucide-react";
import { PromptInput } from "@/components/prompt-input";
import { ChatMessage } from "@/components/chat-message";
import { WorkspacePanel } from "@/components/workspace-panel";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useChatStream } from "@/lib/use-chat-stream";
import { useApi, type DashboardChartPart, type DatasetSummary } from "@/lib/api";

export default function ChatPage() {
  const params = useParams<{ id: string }>();
  const chatId = params.id;
  const router = useRouter();
  const api = useApi();
  const {
    state,
    send,
    stop,
    reset,
    setPinned,
    deleteMessage,
    setFindings,
    patchFinding,
    removeFinding,
  } = useChatStream();
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [dashboardCharts, setDashboardCharts] = useState<DashboardChartPart[]>([]);
  const [workspaceOpen, setWorkspaceOpen] = useState(true);
  const scrollerRef = useRef<HTMLDivElement>(null);

  // Load chat + dataset on mount; reload when the chat id changes.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    (async () => {
      try {
        const chat = await api.getChat(chatId);
        if (cancelled) return;
        const [ds, dashboard] = await Promise.all([
          api.getDataset(chat.datasetId),
          api.getDashboard(chatId),
        ]);
        if (cancelled) return;
        setDataset(ds);
        setDashboardCharts(dashboard);
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

  // Live dashboard updates via DOM event from the SSE consumer.
  useEffect(() => {
    const handler = (e: Event) => {
      const chart = (e as CustomEvent<DashboardChartPart>).detail;
      setDashboardCharts((prev) => {
        if (prev.find((c) => c.id === chart.id)) return prev;
        return [...prev, chart].sort((a, b) => a.position - b.position);
      });
    };
    window.addEventListener("dashboard-chart-added", handler);
    return () => window.removeEventListener("dashboard-chart-added", handler);
  }, []);

  if (loading || !dataset) {
    return (
      <main className="flex flex-1 items-center justify-center text-muted-foreground">
        <Loader2 className="h-5 w-5 animate-spin" />
      </main>
    );
  }

  const busy = state.streaming?.isStreaming ?? false;

  return (
    <main className="flex flex-1 min-w-0">
      <section className="flex flex-col flex-1 min-w-0">
        <div className="flex items-center justify-between gap-4 border-b px-6 py-2 text-sm">
          <div className="flex items-center gap-2 text-muted-foreground min-w-0">
            <Database className="h-4 w-4 shrink-0" />
            <span className="font-medium text-foreground truncate">{dataset.name}</span>
            <Badge variant="secondary" className="shrink-0">
              {dataset.rowCount.toLocaleString()} rows · {dataset.columns.length} cols
            </Badge>
          </div>
          <div className="flex items-center gap-3 shrink-0">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setWorkspaceOpen((v) => !v)}
              aria-label={workspaceOpen ? "Hide workspace" : "Show workspace"}
            >
              {workspaceOpen ? (
                <PanelRightClose className="h-4 w-4" />
              ) : (
                <PanelRightOpen className="h-4 w-4" />
              )}
              <span className="hidden sm:inline">
                {state.findings.length > 0
                  ? `${state.findings.length} findings`
                  : "Workspace"}
              </span>
            </Button>
          </div>
        </div>

        <div ref={scrollerRef} className="flex-1 overflow-y-auto px-6 py-4">
          <div className="max-w-3xl mx-auto">
            {state.messages.length === 0 && !state.streaming ? (
              <div className="text-center text-muted-foreground py-12">
                <p>Ask anything about your data — start with &ldquo;What columns are in this dataset?&rdquo;</p>
              </div>
            ) : null}
            {state.messages
              .filter((m) => m.role === "user" || m.content || m.charts.length > 0 || m.snippets.length > 0 || (m.plotlyCharts?.length ?? 0) > 0)
              .map((m) => (
                <ChatMessage key={m.id} message={m} onPinToggle={setPinned} onDelete={deleteMessage} />
              ))}
            {state.streaming ? <ChatMessage message={state.streaming} /> : null}
          </div>
        </div>

        <div className="border-t px-6 py-3">
          <div className="max-w-3xl mx-auto space-y-2">
            {state.suggestions.length > 0 && !busy ? (
              <div className="flex flex-wrap gap-1.5">
                {state.suggestions.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => { if (chatId) void send(chatId, q); }}
                    className="text-xs px-3 py-1.5 rounded-full border border-border bg-muted/40 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            ) : null}
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
      </section>

      {workspaceOpen ? (
        <WorkspacePanel
          chatId={chatId}
          datasetName={dataset.name}
          findings={state.findings}
          messages={state.messages}
          dashboardCharts={dashboardCharts}
          setDashboardCharts={setDashboardCharts}
          onPatchFinding={patchFinding}
          onRemoveFinding={removeFinding}
          onSetFindings={setFindings}
          className="w-[360px] lg:w-[400px] xl:w-[440px] shrink-0"
        />
      ) : null}
    </main>
  );
}
