"use client";

import { Check, ChevronDown, Copy, Download, Loader2, Pin, PinOff } from "lucide-react";
import dynamic from "next/dynamic";
import type { PlotParams } from "react-plotly.js";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import sql from "highlight.js/lib/languages/sql";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { ChartPart, MessageDTO, PlotlyChartPart, SnippetPart } from "@/lib/api";
import type { StreamingMessage } from "@/lib/use-chat-stream";

hljs.registerLanguage("python", python);
hljs.registerLanguage("sql", sql);

// Dynamic import so Plotly bundle doesn't run during SSR.
const Plot = dynamic<PlotParams>(() => import("@/components/plotly-chart"), { ssr: false });

const SNIPPET_LABEL: Record<SnippetPart["type"], string> = {
  analysis: "📊 Analysis code",
  sql: "🗃️ SQL",
  chart: "🎨 Chart code",
};

function ChartCard({ chart }: { chart: ChartPart }) {
  return (
    <Card className="p-3 mt-3">
      <div className="flex items-center justify-between mb-2">
        {chart.title ? (
          <p className="text-sm font-medium">{chart.title}</p>
        ) : <span />}
        {chart.dataUrl ? (
          <a
            href={chart.dataUrl}
            download={`${chart.title || "chart"}.png`}
            className="text-muted-foreground hover:text-foreground transition-colors p-1 rounded"
            title="Download PNG"
          >
            <Download className="h-3.5 w-3.5" />
          </a>
        ) : null}
      </div>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={chart.dataUrl}
        alt={chart.title || "chart"}
        className="rounded-md border w-full"
      />
    </Card>
  );
}

function PlotlyChartCard({ chart }: { chart: PlotlyChartPart }) {
  let figure: object = {};
  try {
    figure = JSON.parse(chart.spec);
  } catch {
    return <p className="text-sm text-destructive mt-3">Failed to parse chart spec.</p>;
  }
  const fig = figure as { data?: unknown[]; layout?: object };
  return (
    <Card className="p-3 mt-3 overflow-hidden">
      {chart.title ? (
        <p className="text-sm font-medium mb-2">{chart.title}</p>
      ) : null}
      <Plot
        data={(fig.data ?? []) as object[]}
        layout={{
          autosize: true,
          margin: { t: 32, r: 16, b: 48, l: 48 },
          ...fig.layout,
        }}
        config={{ responsive: true, displayModeBar: true, displaylogo: false }}
        style={{ width: "100%", minHeight: 360 }}
        useResizeHandler
      />
    </Card>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const handle = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <button
      onClick={handle}
      className="text-muted-foreground hover:text-foreground transition-colors p-1 rounded"
      title="Copy code"
    >
      {copied ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Copy className="h-3.5 w-3.5" />}
    </button>
  );
}

function SnippetBlock({ snippet }: { snippet: SnippetPart }) {
  const [open, setOpen] = useState(false);
  const language = snippet.type === "sql" ? "sql" : "python";
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (open && codeRef.current && !codeRef.current.dataset.highlighted) {
      hljs.highlightElement(codeRef.current);
    }
  }, [open]);

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mt-3">
      <CollapsibleTrigger className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors">
        <ChevronDown
          className={`h-3 w-3 transition-transform ${open ? "rotate-180" : ""}`}
        />
        {SNIPPET_LABEL[snippet.type]}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-2 rounded-md bg-muted overflow-hidden">
          <div className="flex justify-end px-2 pt-1">
            <CopyButton text={snippet.code} />
          </div>
          <pre className="px-3 pb-3 text-xs overflow-x-auto">
            <code ref={codeRef} className={`language-${language}`}>{snippet.code}</code>
          </pre>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

export function ChatMessage({
  message,
  onPinToggle,
}: {
  message: MessageDTO | StreamingMessage;
  onPinToggle?: (messageId: string, pinned: boolean) => void;
}) {
  const isUser = message.role === "user";
  const isStreaming = "isStreaming" in message && message.isStreaming;
  const toolActive = "toolActive" in message ? message.toolActive : null;
  const error = "error" in message ? message.error : null;
  const pinned = "pinned" in message ? message.pinned : true;
  const canPin = !isUser && !isStreaming && "pinned" in message;
  const plotlyCharts: PlotlyChartPart[] = "plotlyCharts" in message ? (message.plotlyCharts ?? []) : [];

  return (
    <div className={`group flex ${isUser ? "justify-end" : "justify-start"} my-3`}>
      <div
        className={`max-w-[85%] rounded-lg px-4 py-3 ${
          isUser
            ? "bg-primary text-primary-foreground"
            : `bg-card border ${pinned ? "" : "opacity-60"}`
        }`}
      >
        {canPin && onPinToggle ? (
          <div className="float-right -mt-1 -mr-1 ml-2">
            <button
              onClick={() => onPinToggle(message.id, !pinned)}
              className="text-muted-foreground hover:text-primary transition-colors p-1 rounded"
              aria-label={pinned ? "Exclude from report" : "Include in report"}
              title={pinned ? "In report — click to exclude" : "Excluded — click to include"}
            >
              {pinned ? <Pin className="h-3.5 w-3.5" /> : <PinOff className="h-3.5 w-3.5" />}
            </button>
          </div>
        ) : null}
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

        {plotlyCharts.map((pc) => (
          <PlotlyChartCard key={pc.id} chart={pc} />
        ))}

        {message.snippets.map((s) => (
          <SnippetBlock key={s.id} snippet={s} />
        ))}
      </div>
    </div>
  );
}
