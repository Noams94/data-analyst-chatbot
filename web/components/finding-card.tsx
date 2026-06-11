"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { ChevronDown, ChevronUp, FileText, LayoutDashboard, Trash2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import type { FindingPart, PlotlyChartPart } from "@/lib/api";
import { detectHebrew } from "@/lib/i18n";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = {
  finding: FindingPart;
  attachedChart?: PlotlyChartPart | null;
  onTogglePin: (findingId: string, kind: "report" | "dashboard", value: boolean) => void;
  onDelete: (findingId: string) => void;
  onPromoteToDashboard: (findingId: string) => void;
};

const CONFIDENCE_LABELS: Record<string, { en: string; he: string; cls: string }> = {
  high: { en: "High", he: "גבוהה", cls: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400" },
  medium: { en: "Medium", he: "בינונית", cls: "bg-amber-500/15 text-amber-700 dark:text-amber-400" },
  low: { en: "Low", he: "נמוכה", cls: "bg-slate-500/15 text-slate-600 dark:text-slate-400" },
};

export function FindingCard({
  finding,
  attachedChart,
  onTogglePin,
  onDelete,
  onPromoteToDashboard,
}: Props) {
  const [expanded, setExpanded] = useState(false);
  const isHe = detectHebrew(finding.title + " " + finding.claim);
  const conf = CONFIDENCE_LABELS[finding.confidence] ?? CONFIDENCE_LABELS.medium;

  let figure: { data?: Plotly.Data[]; layout?: object } | null = null;
  if (attachedChart) {
    try {
      figure = JSON.parse(attachedChart.spec);
    } catch {
      figure = null;
    }
  }
  const layout = (figure?.layout ?? {}) as Record<string, unknown>;
  // strip any embedded title — card header renders it
  const { title: _t, ...layoutWithoutTitle } = layout as { title?: unknown };
  void _t;

  return (
    <Card className="p-3 gap-2" dir="auto">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold leading-tight" dir="auto">
            {finding.title || (isHe ? "ממצא" : "Finding")}
          </p>
          <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2" dir="auto">
            {finding.claim}
          </p>
        </div>
        <Badge variant="secondary" className={`shrink-0 text-[10px] ${conf.cls}`}>
          {isHe ? conf.he : conf.en}
        </Badge>
      </div>

      {finding.dimension ? (
        <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
          {finding.dimension}
        </p>
      ) : null}

      {figure ? (
        <div className="rounded-md overflow-hidden border bg-card">
          <Plot
            data={figure.data ?? []}
            layout={{
              autosize: true,
              margin: { t: 8, r: 8, b: 36, l: 40 },
              ...layoutWithoutTitle,
              height: 180,
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: "100%", height: 180 }}
            useResizeHandler
          />
        </div>
      ) : null}

      {finding.evidence?.text ? (
        <div>
          <button
            onClick={() => setExpanded((v) => !v)}
            className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
          >
            {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            {isHe ? "פירוט" : "Evidence"}
          </button>
          {expanded ? (
            <p className="text-xs mt-1 text-muted-foreground whitespace-pre-wrap" dir="auto">
              {finding.evidence.text as string}
            </p>
          ) : null}
        </div>
      ) : null}

      <div className="flex items-center gap-1.5 pt-1 border-t">
        <Button
          size="sm"
          variant={finding.includedInReport ? "default" : "outline"}
          className="h-7 text-[11px] flex-1"
          onClick={() => onTogglePin(finding.id, "report", !finding.includedInReport)}
        >
          <FileText className="h-3 w-3" />
          {isHe ? "בדוח" : "In report"}
        </Button>
        <Button
          size="sm"
          variant={finding.includedInDashboard ? "default" : "outline"}
          className="h-7 text-[11px] flex-1"
          onClick={() => {
            if (!finding.includedInDashboard) {
              onPromoteToDashboard(finding.id);
            } else {
              onTogglePin(finding.id, "dashboard", false);
            }
          }}
        >
          <LayoutDashboard className="h-3 w-3" />
          {isHe ? "בדשבורד" : "Dashboard"}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
          onClick={() => onDelete(finding.id)}
          aria-label={isHe ? "מחק" : "Delete"}
        >
          <Trash2 className="h-3 w-3" />
        </Button>
      </div>
    </Card>
  );
}
