/**
 * Convert a ReportDoc into a Hebrew Report Builder import file (.hrb.json).
 *
 * Target schema (from hebrew-report-builder/src/lib/reportFile.js):
 *   {
 *     schema:  "hebrew-report-builder",
 *     version: 3,
 *     exportedAt: <ISO>,
 *     report: {
 *       id, title, blocks: [{ id, type, data: {...} }],
 *       theme: {}, lang: "he"|"en",
 *       createdAt: <number>, updatedAt: <number>
 *     }
 *   }
 *
 * Allowed block types (from ALLOWED_BLOCK_TYPES):
 *   heading | text | image | chart | table | divider | likert | stats | map | cover
 *
 * We use: cover, heading, text, image, divider.
 */

import type { ReportDoc } from "./api";

const SCHEMA = "hebrew-report-builder";
const SCHEMA_VERSION = 3;

type HRBBlock = { id: string; type: string; data: Record<string, unknown> };

function uid(): string {
  // Block ids just need to be unique non-empty strings.
  return crypto.randomUUID();
}

function snippetsToMarkdown(
  snippets: { type: string; code: string }[],
): string {
  if (snippets.length === 0) return "";
  const labels: Record<string, string> = {
    sql: "SQL",
    analysis: "Python (pandas)",
    chart: "Python (matplotlib)",
  };
  const langs: Record<string, string> = {
    sql: "sql",
    analysis: "python",
    chart: "python",
  };
  return snippets
    .map(
      (s) =>
        `**${labels[s.type] ?? s.type}**\n\n\`\`\`${langs[s.type] ?? ""}\n${s.code}\n\`\`\``,
    )
    .join("\n\n");
}

export interface HRBExportOptions {
  title: string;
  subtitle?: string;
  author?: string;
  lang?: "he" | "en";
  includeMethodology?: boolean;
}

export function reportToHRB(report: ReportDoc, opts: HRBExportOptions): unknown {
  const { title, subtitle, author, lang = "en", includeMethodology = true } = opts;

  const blocks: HRBBlock[] = [];

  // 1. Cover block — title + the metadata line.
  const generated = new Date(report.generatedAt).toLocaleDateString(
    lang === "he" ? "he-IL" : undefined,
    { year: "numeric", month: "long", day: "numeric" },
  );
  const datasetLabel = lang === "he" ? "מקור הנתונים" : "Dataset";
  const rowsLabel = lang === "he" ? "שורות" : "rows";
  const colsLabel = lang === "he" ? "עמודות" : "cols";
  const generatedLabel = lang === "he" ? "נוצר" : "Generated";
  const subtitleLine = [
    subtitle,
    `${datasetLabel}: ${report.datasetName} · ${report.datasetMeta.rowCount.toLocaleString()} ${rowsLabel} · ${report.datasetMeta.columns.length} ${colsLabel}`,
    author ? `${author} · ${generatedLabel} ${generated}` : `${generatedLabel} ${generated}`,
  ]
    .filter(Boolean)
    .join("\n\n");

  blocks.push({
    id: uid(),
    type: "cover",
    data: {
      useReportTitle: true,
      overrideTitle: title,
      subtitle: subtitleLine,
    },
  });

  // 2. Sections.
  report.sections.forEach((section, i) => {
    blocks.push({
      id: uid(),
      type: "heading",
      data: { level: 2, text: section.question || `${lang === "he" ? "ממצא" : "Finding"} ${i + 1}` },
    });

    if (section.bodyMd?.trim()) {
      blocks.push({
        id: uid(),
        type: "text",
        data: { markdown: section.bodyMd },
      });
    }

    section.charts.forEach((chart) => {
      blocks.push({
        id: uid(),
        type: "image",
        data: {
          src: chart.dataUrl,
          alt: chart.title || "chart",
          caption: chart.title || "",
        },
      });
    });

    if (includeMethodology && section.snippets.length > 0) {
      blocks.push({
        id: uid(),
        type: "text",
        data: {
          markdown:
            (lang === "he"
              ? "_מתודולוגיה — הקוד שחישב את הממצא:_\n\n"
              : "_Methodology — code used to compute this finding:_\n\n") +
            snippetsToMarkdown(section.snippets),
        },
      });
    }

    if (i < report.sections.length - 1) {
      blocks.push({ id: uid(), type: "divider", data: {} });
    }
  });

  const now = Date.now();
  return {
    schema: SCHEMA,
    version: SCHEMA_VERSION,
    exportedAt: new Date().toISOString(),
    report: {
      id: uid(),
      title,
      blocks,
      theme: {},
      lang,
      createdAt: now,
      updatedAt: now,
    },
  };
}

export function downloadHRB(report: ReportDoc, opts: HRBExportOptions): void {
  const payload = reportToHRB(report, opts);
  const json = JSON.stringify(payload, null, 2);
  const blob = new Blob([json], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const safe = opts.title.replace(/[\\/:*?"<>|]+/g, "-").slice(0, 80) || "report";
  a.href = url;
  a.download = `${safe}.hrb.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
