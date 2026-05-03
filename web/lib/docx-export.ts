/**
 * Client-side Word (.docx) export for ReportDoc.
 * Uses the `docx` library — runs entirely in the browser via Packer.toBlob().
 */
import {
  AlignmentType,
  Document,
  HeadingLevel,
  ImageRun,
  Packer,
  Paragraph,
  TextRun,
} from "docx";
import type { ReportDoc } from "./api";

function dataUrlToUint8Array(dataUrl: string): Uint8Array {
  const base64 = dataUrl.replace(/^data:image\/\w+;base64,/, "");
  const binary = atob(base64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

/** Parse a line of markdown into TextRun[]  (bold, italic, plain). */
function parseInlineRuns(text: string): TextRun[] {
  const runs: TextRun[] = [];
  let i = 0;
  while (i < text.length) {
    const bold = text.indexOf("**", i);
    const italic = text.indexOf("*", i);
    const next = Math.min(
      ...[bold, italic].filter((x) => x !== -1).concat([text.length]),
    );
    if (next > i) {
      runs.push(new TextRun({ text: text.slice(i, next) }));
      i = next;
    }
    if (i >= text.length) break;
    if (text.slice(i, i + 2) === "**") {
      const end = text.indexOf("**", i + 2);
      if (end === -1) { runs.push(new TextRun({ text: text.slice(i) })); break; }
      runs.push(new TextRun({ text: text.slice(i + 2, end), bold: true }));
      i = end + 2;
    } else {
      const end = text.indexOf("*", i + 1);
      if (end === -1) { runs.push(new TextRun({ text: text.slice(i) })); break; }
      runs.push(new TextRun({ text: text.slice(i + 1, end), italics: true }));
      i = end + 1;
    }
  }
  return runs;
}

/** Convert a markdown body to an array of Paragraphs. */
function mdToParagraphs(md: string): Paragraph[] {
  const paragraphs: Paragraph[] = [];
  for (const line of md.split("\n")) {
    const stripped = line.trim();
    if (!stripped) {
      paragraphs.push(new Paragraph({ text: "" }));
      continue;
    }
    // Headings
    const hMatch = stripped.match(/^(#{1,4})\s+(.+)/);
    if (hMatch) {
      const level = [
        HeadingLevel.HEADING_2,
        HeadingLevel.HEADING_3,
        HeadingLevel.HEADING_4,
        HeadingLevel.HEADING_4,
      ][hMatch[1].length - 1];
      paragraphs.push(new Paragraph({ text: hMatch[2], heading: level }));
      continue;
    }
    // Bullet
    const bullet = stripped.match(/^[-*]\s+(.+)/);
    if (bullet) {
      paragraphs.push(
        new Paragraph({ children: parseInlineRuns(bullet[1]), bullet: { level: 0 } }),
      );
      continue;
    }
    paragraphs.push(new Paragraph({ children: parseInlineRuns(stripped) }));
  }
  return paragraphs;
}

export async function downloadDocx(
  report: ReportDoc,
  {
    title,
    subtitle,
    author,
  }: { title: string; subtitle?: string; author?: string },
): Promise<void> {
  const children: Paragraph[] = [];

  // ── Cover info ───────────────────────────────────────────────────────────
  children.push(new Paragraph({ text: title, heading: HeadingLevel.TITLE }));

  if (subtitle) {
    children.push(
      new Paragraph({
        children: [new TextRun({ text: subtitle, italics: true, size: 28 })],
        alignment: AlignmentType.LEFT,
      }),
    );
  }
  if (author) {
    children.push(
      new Paragraph({
        children: [
          new TextRun({ text: "Author: ", bold: true }),
          new TextRun({ text: author }),
        ],
      }),
    );
  }

  children.push(new Paragraph({ text: "" }));
  children.push(
    new Paragraph({
      children: [
        new TextRun({ text: "Dataset: ", bold: true }),
        new TextRun({
          text: `${report.datasetName} · ${report.datasetMeta.rowCount.toLocaleString()} rows · ${report.datasetMeta.columns.length} columns`,
        }),
      ],
    }),
  );
  children.push(
    new Paragraph({
      children: [
        new TextRun({ text: "Generated: ", bold: true }),
        new TextRun({
          text: new Date(report.generatedAt).toLocaleDateString(undefined, {
            year: "numeric",
            month: "long",
            day: "numeric",
          }),
          italics: true,
        }),
      ],
    }),
  );
  children.push(new Paragraph({ text: "" }));

  // ── Sections ──────────────────────────────────────────────────────────────
  for (let i = 0; i < report.sections.length; i++) {
    const sec = report.sections[i];

    children.push(
      new Paragraph({
        text: `${i + 1}. ${sec.question || "Finding"}`,
        heading: HeadingLevel.HEADING_1,
      }),
    );

    children.push(...mdToParagraphs(sec.bodyMd));

    // Embed PNG charts (static ones only — Plotly are interactive, can't embed)
    for (const chart of sec.charts) {
      if (!chart.dataUrl?.startsWith("data:image/png;base64,")) continue;
      try {
        const imageData = dataUrlToUint8Array(chart.dataUrl);
        children.push(new Paragraph({ text: "" }));
        children.push(
          new Paragraph({
            children: [
              new ImageRun({
                data: imageData,
                transformation: { width: 500, height: 280 },
                type: "png",
              }),
            ],
          }),
        );
        if (chart.title) {
          children.push(
            new Paragraph({
              children: [
                new TextRun({ text: chart.title, italics: true, size: 18 }),
              ],
              alignment: AlignmentType.CENTER,
            }),
          );
        }
      } catch {
        // skip unencodable chart
      }
    }

    children.push(new Paragraph({ text: "" }));
  }

  const doc = new Document({ sections: [{ children }] });
  const blob = await Packer.toBlob(doc);

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${title.replace(/[^\w\-]+/g, "_").slice(0, 80) || "report"}.docx`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
