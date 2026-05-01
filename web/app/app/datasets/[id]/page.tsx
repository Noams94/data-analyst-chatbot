"use client";

/**
 * Dataset detail. Three things on this page:
 *   1. Header with the dataset name + counts and a "Start chat" CTA
 *   2. <DatasetPreviewView> — schema, sample rows, numeric summary
 *   3. List of past chats on this dataset
 */

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { Database, Loader2, MessageSquare, Plus } from "lucide-react";
import { DatasetPreviewView } from "@/components/dataset-preview";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  useApi,
  type ChatSummary,
  type DatasetPreview,
  type DatasetSummary,
} from "@/lib/api";

export default function DatasetDetailPage() {
  const params = useParams<{ id: string }>();
  const datasetId = params.id;
  const router = useRouter();
  const api = useApi();
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [creatingChat, setCreatingChat] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([
      api.getDataset(datasetId),
      api.getDatasetPreview(datasetId),
      api.listChats(datasetId),
    ])
      .then(([ds, pv, ch]) => {
        if (cancelled) return;
        setDataset(ds);
        setPreview(pv);
        setChats(ch);
      })
      .catch(() => {
        if (!cancelled) router.replace("/app");
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [api, datasetId, router]);

  const startChat = async () => {
    setCreatingChat(true);
    try {
      const chat = await api.createChat(datasetId);
      router.push(`/app/chat/${chat.id}`);
    } finally {
      setCreatingChat(false);
    }
  };

  if (loading || !dataset || !preview) {
    return (
      <main className="flex flex-1 items-center justify-center text-muted-foreground">
        <Loader2 className="h-5 w-5 animate-spin" />
      </main>
    );
  }

  return (
    <main className="flex-1 overflow-y-auto px-6 py-6 min-w-0">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-muted-foreground text-sm mb-1">
              <Database className="h-3.5 w-3.5" />
              Dataset
            </div>
            <h1 className="text-2xl font-bold truncate">{dataset.name}</h1>
            <div className="flex gap-2 mt-2 text-sm text-muted-foreground">
              <Badge variant="secondary">
                {dataset.rowCount.toLocaleString()} rows
              </Badge>
              <Badge variant="secondary">
                {dataset.columns.length} cols
              </Badge>
              <Badge variant="secondary">
                {(dataset.sizeBytes / 1024).toFixed(1)} KB
              </Badge>
            </div>
          </div>
          <Button onClick={startChat} disabled={creatingChat} size="lg">
            {creatingChat ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Plus className="h-4 w-4" />
            )}
            Start chat
          </Button>
        </div>

        <DatasetPreviewView preview={preview} />

        {/* Chats on this dataset */}
        <section className="border rounded-lg bg-card">
          <div className="px-4 py-2.5 border-b flex items-center justify-between">
            <h2 className="text-sm font-medium">
              Chats on this dataset · {chats.length}
            </h2>
          </div>
          {chats.length === 0 ? (
            <div className="text-muted-foreground text-sm p-6 text-center">
              No chats yet. Click <span className="font-medium">Start chat</span>{" "}
              above to ask your first question.
            </div>
          ) : (
            <ul className="divide-y">
              {chats.map((c) => (
                <li key={c.id}>
                  <Link
                    href={`/app/chat/${c.id}`}
                    className="flex items-center gap-3 px-4 py-2.5 hover:bg-muted/30 transition-colors"
                  >
                    <MessageSquare className="h-4 w-4 text-muted-foreground shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm truncate">
                        {c.title && c.title !== "New chat" ? c.title : "Chat"}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {c.messageCount} message{c.messageCount === 1 ? "" : "s"}{" "}
                        · {new Date(c.updatedAt).toLocaleDateString()}
                      </p>
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </main>
  );
}
