"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { LayoutDashboard, Loader2, Trash2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { toast } from "sonner";
import { useApi, type DashboardChartPart } from "@/lib/api";

// Same SSR-safe dynamic import pattern used in chat-message.tsx
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

function DashboardChartCard({ chart }: { chart: DashboardChartPart }) {
  let figure: { data?: Plotly.Data[]; layout?: object } = {};
  try {
    figure = JSON.parse(chart.spec);
  } catch {
    return (
      <p className="text-sm text-destructive p-4">Failed to parse chart spec.</p>
    );
  }
  return (
    <Card className="p-3 overflow-hidden">
      {chart.title ? (
        <p className="text-sm font-medium mb-2 truncate">{chart.title}</p>
      ) : null}
      <Plot
        data={figure.data ?? []}
        layout={{
          autosize: true,
          margin: { t: 32, r: 16, b: 48, l: 48 },
          ...figure.layout,
        }}
        config={{ responsive: true, displayModeBar: true, displaylogo: false }}
        style={{ width: "100%", minHeight: 300 }}
        useResizeHandler
      />
    </Card>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-muted-foreground gap-3">
      <LayoutDashboard className="h-12 w-12 opacity-30" />
      <p className="text-sm text-center max-w-xs">
        No charts yet — ask the AI to build a dashboard
      </p>
      <p className="text-xs text-center max-w-xs opacity-70">
        Try: &ldquo;Create a dashboard overview of this dataset&rdquo;
      </p>
    </div>
  );
}

export function DashboardDialog({
  chatId,
  open,
  onOpenChange,
  initialCharts,
  onCleared,
}: {
  chatId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  initialCharts?: DashboardChartPart[];
  onCleared?: () => void;
}) {
  const api = useApi();
  const [charts, setCharts] = useState<DashboardChartPart[]>(initialCharts ?? []);
  const [loading, setLoading] = useState(false);
  const [clearing, setClearing] = useState(false);

  useEffect(() => {
    if (!open) return;
    // If caller supplied charts, trust them; otherwise fetch fresh.
    if (initialCharts && initialCharts.length > 0) {
      setCharts(initialCharts);
      return;
    }
    let cancelled = false;
    setLoading(true);
    api
      .getDashboard(chatId)
      .then((data) => { if (!cancelled) setCharts(data); })
      .catch((e: Error) => toast.error(`Failed to load dashboard: ${e.message}`))
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, chatId]);

  // Keep in sync when new charts arrive via SSE while the dialog is open.
  useEffect(() => {
    if (initialCharts) setCharts(initialCharts);
  }, [initialCharts]);

  async function handleClear() {
    setClearing(true);
    try {
      await api.clearDashboard(chatId);
      setCharts([]);
      onCleared?.();
      toast.success("Dashboard cleared");
    } catch (e) {
      toast.error(`Failed to clear: ${(e as Error).message}`);
    } finally {
      setClearing(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-6xl max-h-[90vh] flex flex-col gap-0 p-0">
        <DialogHeader className="px-6 pt-6 pb-4 border-b">
          <DialogTitle className="flex items-center gap-2">
            <LayoutDashboard className="h-4 w-4" />
            Dashboard
            {charts.length > 0 && (
              <Badge variant="secondary">{charts.length}</Badge>
            )}
          </DialogTitle>
          <DialogDescription>
            Charts built by the AI for this chat — interactive, zoomable, and persistent.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto px-6 py-4 min-h-0">
          {loading ? (
            <div className="flex items-center justify-center py-20 text-muted-foreground">
              <Loader2 className="h-5 w-5 animate-spin mr-2" /> Loading…
            </div>
          ) : charts.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {charts.map((chart) => (
                <DashboardChartCard key={chart.id} chart={chart} />
              ))}
            </div>
          )}
        </div>

        <DialogFooter className="px-6 py-4 border-t">
          <Button
            variant="destructive"
            size="sm"
            onClick={handleClear}
            disabled={charts.length === 0 || clearing}
          >
            {clearing ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Trash2 className="h-4 w-4" />
            )}
            Clear dashboard
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
