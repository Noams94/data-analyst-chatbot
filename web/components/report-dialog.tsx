"use client";

import { useEffect, useRef, useState } from "react";
import { Download, FileText, Loader2, Send } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import { useApi, type ReportDoc } from "@/lib/api";
import { downloadHRB } from "@/lib/hrb-export";
import { downloadDocx } from "@/lib/docx-export";
import { ReportView, REPORT_STYLES, reportToStandaloneHTML } from "./report-view";

export function ReportDialog({
  chatId,
  open,
  onOpenChange,
  defaultTitle,
}: {
  chatId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  defaultTitle: string;
}) {
  const api = useApi();
  const [report, setReport] = useState<ReportDoc | null>(null);
  const [loading, setLoading] = useState(false);
  const [title, setTitle] = useState(defaultTitle);
  const [subtitle, setSubtitle] = useState("");
  const [author, setAuthor] = useState("");
  const previewRef = useRef<HTMLDivElement>(null);

  // Fetch the latest report whenever the dialog opens.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    setReport(null);
    api
      .getReport(chatId)
      .then((doc) => {
        if (cancelled) return;
        setReport(doc);
        if (!title) setTitle(`${doc.datasetName} — Analysis report`);
      })
      .catch((e: Error) => {
        toast.error(`Failed to build report: ${e.message}`);
        onOpenChange(false);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, chatId]);

  function downloadHTML() {
    if (!report || !previewRef.current) return;
    // The on-screen preview is rendered with Tailwind/shadcn classes, but the
    // standalone download uses the inline REPORT_STYLES so it's portable.
    const article = previewRef.current.querySelector(".report-doc");
    if (!article) return;
    const html = reportToStandaloneHTML(article.outerHTML, title);
    const blob = new Blob([html], { type: "text/html;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${title.replace(/[^\w\-]+/g, "_").slice(0, 80) || "report"}.html`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toast.success("Report downloaded");
  }

  function downloadMarkdown() {
    if (!report) return;
    const lines: string[] = [];
    lines.push(`# ${title}`);
    if (subtitle) lines.push(`\n_${subtitle}_`);
    if (author) lines.push(`\n**Author:** ${author}`);
    lines.push(`\n**Dataset:** ${report.datasetName} · ${report.datasetMeta.rowCount.toLocaleString()} rows · ${report.datasetMeta.columns.length} columns`);
    lines.push(`\n**Generated:** ${new Date(report.generatedAt).toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" })}`);
    lines.push("\n---");
    if (report.sections.length === 0) {
      lines.push("\n_No findings pinned._");
    } else {
      report.sections.forEach((s, i) => {
        lines.push(`\n## ${i + 1}. ${s.question || "Finding"}`);
        lines.push(`\n${s.bodyMd}`);
        if (s.snippets.length > 0) {
          lines.push("\n<details>\n<summary>Code</summary>\n");
          s.snippets.forEach((sn) => {
            const lang = sn.type === "sql" ? "sql" : "python";
            lines.push(`\`\`\`${lang}\n${sn.code}\n\`\`\``);
          });
          lines.push("</details>");
        }
      });
    }
    const md = lines.join("\n");
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${title.replace(/[^\w\-]+/g, "_").slice(0, 80) || "report"}.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toast.success("Markdown downloaded");
  }

  function copyJSON() {
    if (!report) return;
    const payload = { title, subtitle, author, ...report };
    navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    toast.success("ReportDoc JSON copied to clipboard");
  }

  async function downloadWord() {
    if (!report) return;
    await downloadDocx(report, { title, subtitle, author });
    toast.success("Word document downloaded");
  }

  function downloadForReportBuilder() {
    if (!report) return;
    downloadHRB(report, { title, subtitle, author, lang: "en" });
    toast.success(
      "Saved .hrb.json — open reports.noamkeshet.com → Import → choose this file",
    );
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Generate report</DialogTitle>
          <DialogDescription>
            Curated, publishable summary of this analysis. Pinned assistant
            turns become numbered sections.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3 grid-cols-1 sm:grid-cols-3">
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">Title</label>
            <Input value={title} onChange={(e) => setTitle(e.target.value)} />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">Subtitle (optional)</label>
            <Input value={subtitle} onChange={(e) => setSubtitle(e.target.value)} />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">Author (optional)</label>
            <Input value={author} onChange={(e) => setAuthor(e.target.value)} />
          </div>
        </div>

        <div className="rounded-md border bg-muted/30 p-2 max-h-[55vh] overflow-y-auto">
          {/* Inline the report styles so the on-screen preview matches the
              downloaded HTML one-to-one. */}
          <style dangerouslySetInnerHTML={{ __html: REPORT_STYLES }} />
          <div ref={previewRef}>
            {loading ? (
              <div className="flex items-center justify-center py-16 text-muted-foreground">
                <Loader2 className="h-5 w-5 animate-spin mr-2" /> Building…
              </div>
            ) : report ? (
              <ReportView
                report={report}
                title={title}
                subtitle={subtitle}
                author={author}
              />
            ) : null}
          </div>
        </div>

        <DialogFooter className="gap-2 flex-wrap">
          <Button variant="outline" onClick={copyJSON} disabled={!report}>
            <FileText className="h-4 w-4" /> Copy JSON
          </Button>
          <Button variant="outline" onClick={downloadMarkdown} disabled={!report}>
            <Download className="h-4 w-4" /> Download Markdown
          </Button>
          <Button variant="outline" onClick={downloadHTML} disabled={!report}>
            <Download className="h-4 w-4" /> Download HTML
          </Button>
          <Button variant="outline" onClick={downloadWord} disabled={!report}>
            <Download className="h-4 w-4" /> Download Word
          </Button>
          <Button onClick={downloadForReportBuilder} disabled={!report}>
            <Send className="h-4 w-4" /> Export for Report Builder
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
