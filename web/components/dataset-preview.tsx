"use client";

/**
 * Three-section preview of a dataset:
 *   1. Schema  — every column with dtype, null %, unique count, sample values
 *   2. Sample  — first 5 rows
 *   3. Numeric — describe() summary (mean / std / min / quartiles / max)
 */

import { useMemo, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { ColumnInfo, DatasetPreview } from "@/lib/api";

const NUMERIC_STAT_LABELS: Array<[string, string]> = [
  ["mean", "mean"],
  ["std", "std"],
  ["min", "min"],
  ["25%", "25%"],
  ["50%", "median"],
  ["75%", "75%"],
  ["max", "max"],
];

function formatNum(v: number | null | undefined): string {
  if (v == null) return "—";
  if (Number.isInteger(v)) return v.toLocaleString();
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return v.toLocaleString(undefined, { maximumSignificantDigits: 4 });
}

function CellValue({ v }: { v: unknown }) {
  if (v == null) return <span className="text-muted-foreground italic">null</span>;
  if (typeof v === "number") return <>{formatNum(v)}</>;
  return <>{String(v)}</>;
}

function Section({
  title,
  defaultOpen = true,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="border rounded-lg bg-card">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-4 py-2.5 text-sm font-medium hover:bg-muted/30 transition-colors"
      >
        {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        {title}
      </button>
      {open ? <div className="border-t">{children}</div> : null}
    </section>
  );
}

export function DatasetPreviewView({ preview }: { preview: DatasetPreview }) {
  const sampleColumns = useMemo(
    () => preview.columns.map((c) => c.name),
    [preview.columns],
  );
  const numericCols = Object.keys(preview.numericSummary ?? {});

  return (
    <div className="space-y-4">
      {/* Schema */}
      <Section title={`Schema · ${preview.columns.length} columns`}>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted/30 text-xs text-muted-foreground uppercase tracking-wide">
              <tr>
                <th className="text-left px-4 py-2 font-medium">Column</th>
                <th className="text-left px-4 py-2 font-medium">Type</th>
                <th className="text-right px-4 py-2 font-medium">Null %</th>
                <th className="text-right px-4 py-2 font-medium">Unique</th>
                <th className="text-left px-4 py-2 font-medium">Sample values</th>
              </tr>
            </thead>
            <tbody>
              {preview.columns.map((c: ColumnInfo) => (
                <tr key={c.name} className="border-t">
                  <td className="px-4 py-1.5 font-medium">{c.name}</td>
                  <td className="px-4 py-1.5 text-muted-foreground">{c.dtype}</td>
                  <td className="px-4 py-1.5 text-right tabular-nums text-muted-foreground">
                    {c.nullPct.toFixed(1)}%
                  </td>
                  <td className="px-4 py-1.5 text-right tabular-nums">
                    {c.uniqueCount.toLocaleString()}
                  </td>
                  <td className="px-4 py-1.5 text-muted-foreground truncate max-w-[280px]">
                    {c.sampleValues.join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* Sample rows */}
      {preview.sampleRows.length > 0 ? (
        <Section title={`Sample · first ${preview.sampleRows.length} rows`}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted/30 text-xs text-muted-foreground uppercase tracking-wide">
                <tr>
                  {sampleColumns.map((col) => (
                    <th key={col} className="text-left px-3 py-2 font-medium whitespace-nowrap">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.sampleRows.map((row, i) => (
                  <tr key={i} className="border-t">
                    {sampleColumns.map((col) => (
                      <td
                        key={col}
                        className="px-3 py-1.5 whitespace-nowrap max-w-[280px] truncate"
                      >
                        <CellValue v={row[col]} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      ) : null}

      {/* Numeric summary */}
      {numericCols.length > 0 ? (
        <Section title={`Numeric summary · ${numericCols.length} columns`} defaultOpen={false}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted/30 text-xs text-muted-foreground uppercase tracking-wide">
                <tr>
                  <th className="text-left px-4 py-2 font-medium">Column</th>
                  {NUMERIC_STAT_LABELS.map(([_, label]) => (
                    <th key={label} className="text-right px-3 py-2 font-medium tabular-nums">
                      {label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {numericCols.map((col) => (
                  <tr key={col} className="border-t">
                    <td className="px-4 py-1.5 font-medium whitespace-nowrap">{col}</td>
                    {NUMERIC_STAT_LABELS.map(([key, label]) => (
                      <td key={label} className="px-3 py-1.5 text-right tabular-nums">
                        {formatNum(preview.numericSummary[col]?.[key])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      ) : null}
    </div>
  );
}
