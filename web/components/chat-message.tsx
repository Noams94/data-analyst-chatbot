"use client";

import { ChevronDown, Loader2 } from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { ChartPart, MessageDTO, SnippetPart } from "@/lib/api";
import type { StreamingMessage } from "@/lib/use-chat-stream";
import { api } from "@/lib/api";

const SNIPPET_LABEL: Record<SnippetPart["type"], string> = {
  analysis: "📊 Analysis code",
  sql: "🗃️ SQL",
  chart: "🎨 Chart code",
};

function ChartCard({ chart }: { chart: ChartPart }) {
  return (
    <Card className="p-3 mt-3">
      {chart.title ? (
        <p className="text-sm font-medium mb-2">{chart.title}</p>
      ) : null}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={api.chartUrl(chart.url)}
        alt={chart.title || "chart"}
        className="rounded-md border w-full"
      />
    </Card>
  );
}

function SnippetBlock({ snippet }: { snippet: SnippetPart }) {
  const [open, setOpen] = useState(false);
  const language = snippet.type === "sql" ? "sql" : "python";
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mt-3">
      <CollapsibleTrigger className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors">
        <ChevronDown
          className={`h-3 w-3 transition-transform ${open ? "rotate-180" : ""}`}
        />
        {SNIPPET_LABEL[snippet.type]}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <pre className="mt-2 rounded-md bg-muted p-3 text-xs overflow-x-auto">
          <code className={`language-${language}`}>{snippet.code}</code>
        </pre>
      </CollapsibleContent>
    </Collapsible>
  );
}

export function ChatMessage({
  message,
}: {
  message: MessageDTO | StreamingMessage;
}) {
  const isUser = message.role === "user";
  const isStreaming = "isStreaming" in message && message.isStreaming;
  const toolActive = "toolActive" in message ? message.toolActive : null;
  const error = "error" in message ? message.error : null;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} my-3`}>
      <div
        className={`max-w-[85%] rounded-lg px-4 py-3 ${
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-card border"
        }`}
      >
        {message.content ? (
          isUser ? (
            <div className="whitespace-pre-wrap text-sm leading-relaxed">
              {message.content}
            </div>
          ) : (
            <div className="prose prose-sm max-w-none text-sm leading-relaxed prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-headings:my-3 prose-pre:bg-muted prose-pre:text-foreground">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
              {isStreaming ? (
                <span className="ml-0.5 inline-block w-1.5 h-4 bg-current animate-pulse align-text-bottom" />
              ) : null}
            </div>
          )
        ) : isStreaming ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" />
            Thinking…
          </div>
        ) : null}

        {toolActive ? (
          <Badge variant="secondary" className="mt-2">
            <Loader2 className="h-3 w-3 animate-spin" /> {toolActive}
          </Badge>
        ) : null}

        {error ? (
          <p className="mt-2 text-sm text-destructive">⚠️ {error}</p>
        ) : null}

        {message.charts.map((c) => (
          <ChartCard key={c.id} chart={c} />
        ))}

        {message.snippets.map((s) => (
          <SnippetBlock key={s.id} snippet={s} />
        ))}
      </div>
    </div>
  );
}
