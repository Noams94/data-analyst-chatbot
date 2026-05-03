"use client";

import { useCallback, useRef, useState } from "react";
import { Upload, Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { useApi, type DatasetSummary } from "@/lib/api";

export function UploadDropzone({ onUploaded }: { onUploaded: (ds: DatasetSummary) => void }) {
  const api = useApi();
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);

  const upload = useCallback(
    async (file: File) => {
      setBusy(true);
      setErr(null);
      try {
        const ds = await api.uploadDataset(file);
        onUploaded(ds);
      } catch (e) {
        setErr((e as Error).message || "Upload failed");
      } finally {
        setBusy(false);
      }
    },
    [api, onUploaded],
  );

  return (
    <Card
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        const file = e.dataTransfer.files?.[0];
        if (file) void upload(file);
      }}
      className={`flex flex-col items-center gap-4 p-12 text-center max-w-md cursor-pointer transition-colors ${
        drag ? "border-primary bg-primary/5" : ""
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.tsv,.xlsx,.xls,.json,.parquet"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void upload(file);
        }}
      />
      <div className="rounded-full bg-primary/10 p-4">
        {busy ? (
          <Loader2 className="h-8 w-8 text-primary animate-spin" />
        ) : (
          <Upload className="h-8 w-8 text-primary" />
        )}
      </div>
      <h2 className="text-xl font-semibold">
        {busy ? "Uploading…" : "Drop a dataset to start"}
      </h2>
      <p className="text-sm text-muted-foreground">
        CSV, TSV, Excel, JSON, or Parquet up to 50&nbsp;MB.
      </p>
      {err ? <p className="text-sm text-destructive">{err}</p> : null}
    </Card>
  );
}
