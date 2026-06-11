"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  FileText,
  LayoutDashboard,
  Lightbulb,
  Loader2,
  Pencil,
  Sparkles,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { FindingCard } from "@/components/finding-card";
import { ReportDialog } from "@/components/report-dialog";
import {
  useApi,
  type DashboardChartPart,
  type FindingPart,
  type MessageDTO,
  type PlotlyChartPart,
} from "@/lib/api";
import { detectHebrew, makeT, type UiLang } from "@/lib/i18n";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type TabId = "findings" | "dashboard" | "report";

function flattenPlotlyCharts(messages: MessageDTO[]): Map<string, PlotlyChartPart> {
  const m = new Map<string, PlotlyChartPart>();
  for (const msg of messages) {
    for (const c of msg.plotlyCharts ?? []) {
      m.set(c.id, c);
    }
  }
  return m;
}

export function WorkspacePanel({
  chatId,
  datasetName,
  findings,
  messages,
  dashboardCharts,
  setDashboardCharts,
  onPatchFinding,
  onRemoveFinding,
  onSetFindings,
  className,
  initialTab = "findings",
  uiLanguage = "auto",
}: {
  chatId: string;
  datasetName: string;
  findings: FindingPart[];
  messages: MessageDTO[];
  dashboardCharts: DashboardChartPart[];
  setDashboardCharts: React.Dispatch<React.SetStateAction<DashboardChartPart[]>>;
  onPatchFinding: (
    findingId: string,
    updates: Partial<{
      title: string;
      claim: string;
      confidence: "low" | "medium" | "high";
      dimension: string;
      includedInReport: boolean;
      includedInDashboard: boolean;
      position: number;
    }>,
  ) => void | Promise<void>;
  onRemoveFinding: (findingId: string) => void | Promise<void>;
  onSetFindings: (findings: FindingPart[]) => void;
  className?: string;
  initialTab?: TabId;
  uiLanguage?: "en" | "he" | "auto";
}) {
  const api = useApi();
  const [tab, setTab] = useState<TabId>(initialTab);
  const [reportOpen, setReportOpen] = useState(false);
  const [loadingFindings, setLoadingFindings] = useState(false);
  const [filter, setFilter] = useState<"all" | "report" | "dashboard">("all");

  const lang: UiLang =
    uiLanguage !== "auto"
      ? uiLanguage
      : findings.some((f) => detectHebrew(f.title + " " + f.claim))
      ? "he"
      : detectHebrew(datasetName)
      ? "he"
      : "en";
  const t = makeT(lang);

  // Initial fetch of findings list — covers the case where the user reloads
  // the page mid-session and we need to populate the workspace.
  useEffect(() => {
    let cancelled = false;
    setLoadingFindings(true);
    api
      .listFindings(chatId)
      .then((rows) => {
        if (!cancelled) onSetFindings(rows);
      })
      .catch(() => {
        if (!cancelled) onSetFindings([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingFindings(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId]);

  const charts = useMemo(() => flattenPlotlyCharts(messages), [messages]);

  const visibleFindings = findings.filter((f) => {
    if (filter === "report") return f.includedInReport;
    if (filter === "dashboard") return f.includedInDashboard;
    return true;
  });

  async function handlePromoteToDashboard(findingId: string) {
    try {
      const newChart = await api.promoteFindingToDashboard(chatId, findingId);
      setDashboardCharts((prev) =>
        [...prev.filter((c) => c.id !== newChart.id), newChart].sort(
          (a, b) => a.position - b.position,
        ),
      );
      onPatchFinding(findingId, { includedInDashboard: true });
      toast.success(t("Added to dashboard", "נוסף לדשבורד"));
    } catch (e) {
      toast.error((e as Error).message);
    }
  }

  function handleTogglePin(
    findingId: string,
    kind: "report" | "dashboard",
    value: boolean,
  ) {
    if (kind === "report") {
      onPatchFinding(findingId, { includedInReport: value });
    } else if (kind === "dashboard" && !value) {
      // The PATCH below also deletes the promoted chart(s) server-side;
      // mirror that locally so the Dashboard tab stays in sync.
      onPatchFinding(findingId, { includedInDashboard: false });
      setDashboardCharts((prev) => prev.filter((c) => c.findingId !== findingId));
    }
  }

  return (
    <aside
      className={`flex flex-col border-l bg-background ${className ?? ""}`}
      dir="auto"
    >
      <div className="flex items-center gap-1 border-b px-2 py-1.5">
        {([
          ["findings", Lightbulb, t("Findings", "ממצאים"), findings.length],
          [
            "dashboard",
            LayoutDashboard,
            t("Dashboard", "דשבורד"),
            dashboardCharts.length,
          ],
          [
            "report",
            FileText,
            t("Report", "דוח"),
            findings.filter((f) => f.includedInReport).length,
          ],
        ] as const).map(([id, Icon, label, count]) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-md transition-colors ${
              tab === id
                ? "bg-muted text-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Icon className="h-3.5 w-3.5" />
            {label}
            {count > 0 ? (
              <Badge variant="secondary" className="h-4 px-1 text-[10px]">
                {count}
              </Badge>
            ) : null}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto min-h-0">
        {tab === "findings" ? (
          <FindingsTab
            findings={visibleFindings}
            allFindings={findings}
            loading={loadingFindings}
            charts={charts}
            filter={filter}
            setFilter={setFilter}
            onTogglePin={handleTogglePin}
            onDelete={onRemoveFinding}
            onPromoteToDashboard={handlePromoteToDashboard}
            lang={lang}
          />
        ) : null}

        {tab === "dashboard" ? (
          <DashboardTab
            chatId={chatId}
            charts={dashboardCharts}
            setCharts={setDashboardCharts}
            lang={lang}
          />
        ) : null}

        {tab === "report" ? (
          <ReportTab
            findings={findings.filter((f) => f.includedInReport)}
            charts={charts}
            onOpen={() => setReportOpen(true)}
            lang={lang}
          />
        ) : null}
      </div>

      <ReportDialog
        chatId={chatId}
        open={reportOpen}
        onOpenChange={setReportOpen}
        defaultTitle={`${datasetName} — ${t("Analysis report", "דוח ניתוח")}`}
      />
    </aside>
  );
}

// ─── Findings tab ────────────────────────────────────────────────────────────

function FindingsTab({
  findings,
  allFindings,
  loading,
  charts,
  filter,
  setFilter,
  onTogglePin,
  onDelete,
  onPromoteToDashboard,
  lang,
}: {
  findings: FindingPart[];
  allFindings: FindingPart[];
  loading: boolean;
  charts: Map<string, PlotlyChartPart>;
  filter: "all" | "report" | "dashboard";
  setFilter: (v: "all" | "report" | "dashboard") => void;
  onTogglePin: (id: string, kind: "report" | "dashboard", value: boolean) => void;
  onDelete: (id: string) => void | Promise<void>;
  onPromoteToDashboard: (id: string) => void;
  lang: "en" | "he";
}) {
  const t = makeT(lang);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
      </div>
    );
  }

  return (
    <div className="p-3 space-y-2.5">
      <div className="flex items-center gap-1 text-xs">
        {([
          ["all", t("All", "הכל"), allFindings.length],
          ["report", t("In report", "בדוח"), allFindings.filter((f) => f.includedInReport).length],
          ["dashboard", t("In dashboard", "בדשבורד"), allFindings.filter((f) => f.includedInDashboard).length],
        ] as const).map(([id, label, count]) => (
          <button
            key={id}
            onClick={() => setFilter(id)}
            className={`px-2 py-0.5 rounded-full border ${
              filter === id
                ? "bg-foreground text-background border-foreground"
                : "border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            {label} {count > 0 ? `· ${count}` : ""}
          </button>
        ))}
      </div>

      {findings.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-muted-foreground gap-2">
          <Sparkles className="h-8 w-8 opacity-30" />
          <p className="text-xs text-center max-w-[200px]">
            {t(
              "Findings will appear here as the AI analyzes your data.",
              "ממצאים יופיעו כאן ככל שה-AI ינתח את הנתונים שלך.",
            )}
          </p>
        </div>
      ) : (
        findings.map((f) => (
          <FindingCard
            key={f.id}
            finding={f}
            attachedChart={f.chartId ? charts.get(f.chartId) ?? null : null}
            onTogglePin={onTogglePin}
            onDelete={onDelete}
            onPromoteToDashboard={onPromoteToDashboard}
          />
        ))
      )}
    </div>
  );
}

// ─── Dashboard tab ───────────────────────────────────────────────────────────

function DashboardChartCell({
  chart,
  onPatch,
  onDelete,
  onMove,
  isFirst,
  isLast,
  lang,
}: {
  chart: DashboardChartPart;
  onPatch: (chartId: string, updates: Partial<{ title: string; layoutW: number; layoutH: number }>) => void;
  onDelete: (chartId: string) => void;
  onMove: (chartId: string, direction: "up" | "down") => void;
  isFirst: boolean;
  isLast: boolean;
  lang: "en" | "he";
}) {
  const [editing, setEditing] = useState(false);
  const [titleDraft, setTitleDraft] = useState(chart.title);
  const t = makeT(lang);

  let figure: { data?: Plotly.Data[]; layout?: object } = {};
  try {
    figure = JSON.parse(chart.spec);
  } catch {
    return <p className="text-xs text-destructive">Bad spec</p>;
  }
  const layout = (figure.layout ?? {}) as Record<string, unknown>;
  const { title: _ignored, ...layoutWithoutTitle } = layout as { title?: unknown };
  void _ignored;

  const w = chart.layoutW ?? 6;

  return (
    <Card className={`p-3 gap-2 ${w >= 9 ? "col-span-2" : ""}`} dir="auto">
      <div className="flex items-start gap-2">
        {editing ? (
          <div className="flex-1 flex gap-1">
            <Input
              autoFocus
              value={titleDraft}
              onChange={(e) => setTitleDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  onPatch(chart.id, { title: titleDraft });
                  setEditing(false);
                }
                if (e.key === "Escape") setEditing(false);
              }}
              className="h-7 text-sm"
              dir="auto"
            />
            <Button
              size="sm"
              variant="outline"
              className="h-7"
              onClick={() => {
                onPatch(chart.id, { title: titleDraft });
                setEditing(false);
              }}
            >
              {t("Save", "שמור")}
            </Button>
          </div>
        ) : (
          <p
            className="text-sm font-semibold flex-1 truncate cursor-pointer hover:text-primary"
            dir="auto"
            onClick={() => {
              setTitleDraft(chart.title);
              setEditing(true);
            }}
          >
            {chart.title || t("Untitled", "ללא כותרת")}
            <Pencil className="inline h-3 w-3 ms-1 opacity-40" />
          </p>
        )}
        <div className="flex items-center gap-0.5 shrink-0">
          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0"
            onClick={() => onMove(chart.id, "up")}
            disabled={isFirst}
            aria-label={t("Move up", "העבר למעלה")}
          >
            <ArrowUp className="h-3 w-3" />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0"
            onClick={() => onMove(chart.id, "down")}
            disabled={isLast}
            aria-label={t("Move down", "העבר למטה")}
          >
            <ArrowDown className="h-3 w-3" />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0"
            onClick={() =>
              onPatch(chart.id, { layoutW: w >= 9 ? 6 : 12 })
            }
            aria-label={t("Toggle width", "החלף רוחב")}
            title={t("Wide / narrow", "רחב / צר")}
          >
            <span className="text-[10px] font-mono">{w >= 9 ? "1×" : "2×"}</span>
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"
            onClick={() => onDelete(chart.id)}
            aria-label={t("Delete", "מחק")}
          >
            <Trash2 className="h-3 w-3" />
          </Button>
        </div>
      </div>
      <Plot
        data={figure.data ?? []}
        layout={{
          autosize: true,
          margin: { t: 10, r: 12, b: 40, l: 44 },
          ...layoutWithoutTitle,
        }}
        config={{ responsive: true, displayModeBar: "hover", displaylogo: false }}
        style={{ width: "100%", height: 280 }}
        useResizeHandler
      />
    </Card>
  );
}

function DashboardTab({
  chatId,
  charts,
  setCharts,
  lang,
}: {
  chatId: string;
  charts: DashboardChartPart[];
  setCharts: React.Dispatch<React.SetStateAction<DashboardChartPart[]>>;
  lang: UiLang;
}) {
  const api = useApi();
  const t = makeT(lang);
  const [clearing, setClearing] = useState(false);

  // Functional updates throughout — rapid consecutive actions must not
  // clobber each other through a stale `charts` closure.
  async function patch(
    chartId: string,
    updates: Partial<{ title: string; layoutW: number; layoutH: number }>,
  ) {
    setCharts((prev) => prev.map((c) => (c.id === chartId ? { ...c, ...updates } : c)));
    try {
      await api.patchDashboardChart(chatId, chartId, updates);
    } catch (e) {
      toast.error((e as Error).message);
    }
  }

  async function remove(chartId: string) {
    setCharts((prev) => prev.filter((c) => c.id !== chartId));
    try {
      await api.deleteDashboardChart(chatId, chartId);
    } catch (e) {
      toast.error((e as Error).message);
    }
  }

  async function move(chartId: string, direction: "up" | "down") {
    const idx = charts.findIndex((c) => c.id === chartId);
    const target = direction === "up" ? idx - 1 : idx + 1;
    if (idx < 0 || target < 0 || target >= charts.length) return;
    const reordered = [...charts];
    [reordered[idx], reordered[target]] = [reordered[target], reordered[idx]];
    const withPos = reordered.map((c, i) => ({ ...c, position: i }));
    setCharts(withPos);
    try {
      await api.reorderDashboard(chatId, withPos.map((c) => c.id));
    } catch (e) {
      toast.error((e as Error).message);
    }
  }

  async function clearAll() {
    if (!confirm(t("Clear all dashboard charts?", "לנקות את כל גרפי הדשבורד?"))) return;
    setClearing(true);
    try {
      await api.clearDashboard(chatId);
      setCharts([]);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setClearing(false);
    }
  }

  if (charts.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-muted-foreground gap-2 px-6">
        <LayoutDashboard className="h-10 w-10 opacity-30" />
        <p className="text-xs text-center max-w-[240px]">
          {t(
            'Ask the AI to "build a dashboard" or click "In dashboard" on any finding.',
            'בקש מה-AI "לבנות דשבורד" או לחץ "בדשבורד" בכרטיס ממצא.',
          )}
        </p>
      </div>
    );
  }

  return (
    <div className="p-3 space-y-3">
      <div className="grid grid-cols-2 gap-3">
        {charts.map((chart, i) => (
          <DashboardChartCell
            key={chart.id}
            chart={chart}
            onPatch={patch}
            onDelete={remove}
            onMove={move}
            isFirst={i === 0}
            isLast={i === charts.length - 1}
            lang={lang}
          />
        ))}
      </div>
      <div className="pt-2 border-t flex items-center justify-end">
        <Button
          variant="ghost"
          size="sm"
          onClick={clearAll}
          disabled={clearing}
          className="text-muted-foreground hover:text-destructive"
        >
          {clearing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
          {t("Clear all", "נקה הכל")}
        </Button>
      </div>
    </div>
  );
}

// ─── Report tab ──────────────────────────────────────────────────────────────

function ReportTab({
  findings,
  charts,
  onOpen,
  lang,
}: {
  findings: FindingPart[];
  charts: Map<string, PlotlyChartPart>;
  onOpen: () => void;
  lang: "en" | "he";
}) {
  const t = makeT(lang);
  return (
    <div className="p-3 space-y-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs text-muted-foreground">
          {t(
            "Findings flagged for the report:",
            "ממצאים שסומנו לדוח:",
          )}{" "}
          <span className="text-foreground font-medium">{findings.length}</span>
        </p>
        <Button size="sm" variant="default" onClick={onOpen}>
          <FileText className="h-3.5 w-3.5" />
          {t("Open report builder", "פתח בונה דוחות")}
        </Button>
      </div>
      {findings.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-muted-foreground gap-2">
          <FileText className="h-8 w-8 opacity-30" />
          <p className="text-xs text-center max-w-[220px]">
            {t(
              "Toggle 'In report' on findings to add them here. Open the report builder to export to DOCX/HTML.",
              "סמן 'בדוח' בממצאים כדי להוסיף אותם. פתח את בונה הדוחות לייצוא ל-DOCX/HTML.",
            )}
          </p>
        </div>
      ) : (
        <ul className="space-y-2">
          {findings.map((f, i) => (
            <li key={f.id} className="border rounded-md p-2.5 space-y-1.5 bg-card" dir="auto">
              <div className="flex items-start gap-2">
                <span className="text-xs text-muted-foreground tabular-nums w-5 shrink-0">
                  {i + 1}.
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium leading-tight" dir="auto">
                    {f.title}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5" dir="auto">
                    {f.claim}
                  </p>
                </div>
              </div>
              {f.chartId && charts.get(f.chartId) ? (
                <p className="text-[10px] text-muted-foreground ms-7">
                  📊 {charts.get(f.chartId)?.title}
                </p>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
