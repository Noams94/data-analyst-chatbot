"use client";

/**
 * Left-rail navigation. Lists the user's datasets (top section) and recent
 * chats (bottom section), with a "+ New" button that drops the user back to
 * the landing page so they can upload or pick an existing dataset.
 *
 * Both lists are loaded in parallel. We poll on `pathname` change so adding
 * a new chat or dataset shows up immediately without a manual refetch.
 */

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Database,
  MessageSquare,
  Plus,
  Loader2,
  PanelLeftClose,
  PanelLeftOpen,
  Settings,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useApi, type ChatSummary, type DatasetSummary } from "@/lib/api";

function timeAgo(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d`;
  return d.toLocaleDateString();
}

export function Sidebar() {
  const api = useApi();
  const pathname = usePathname();
  const router = useRouter();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [collapsed, setCollapsed] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  // Refetch when the route changes — covers "user just created a chat" without
  // wiring a global state store.
  const cacheKey = pathname;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([api.listDatasets(), api.listChats()])
      .then(([ds, ch]) => {
        if (cancelled) return;
        setDatasets(ds);
        setChats(ch);
      })
      .catch(() => {
        // Silent — sidebar is non-critical; main view shows real errors.
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [api, cacheKey]);

  // Update chat title in-place when the streaming assistant emits a `title` event.
  useEffect(() => {
    const handler = (e: Event) => {
      const { chatId, title } = (e as CustomEvent<{ chatId: string; title: string }>).detail;
      setChats((prev) =>
        prev.map((c) => (c.id === chatId ? { ...c, title } : c))
      );
    };
    window.addEventListener("chat-title-changed", handler);
    return () => window.removeEventListener("chat-title-changed", handler);
  }, []);

  const activeChatId = pathname?.startsWith("/app/chat/")
    ? pathname.split("/")[3]
    : null;
  const activeDatasetId = pathname?.startsWith("/app/datasets/")
    ? pathname.split("/")[3]
    : null;

  const handleDeleteDataset = async (id: string) => {
    if (confirmDelete !== id) { setConfirmDelete(id); return; }
    setConfirmDelete(null);
    setDatasets((prev) => prev.filter((d) => d.id !== id));
    if (activeDatasetId === id) router.push("/app");
    try { await api.deleteDataset(id); } catch { /* already removed from UI */ }
  };

  const handleDeleteChat = async (id: string) => {
    if (confirmDelete !== id) { setConfirmDelete(id); return; }
    setConfirmDelete(null);
    setChats((prev) => prev.filter((c) => c.id !== id));
    if (activeChatId === id) router.push("/app");
    try { await api.deleteChat(id); } catch { /* already removed from UI */ }
  };

  if (collapsed) {
    return (
      <aside className="w-12 shrink-0 border-r flex flex-col items-center py-3">
        <button
          aria-label="Expand sidebar"
          className="text-muted-foreground hover:text-foreground p-1"
          onClick={() => setCollapsed(false)}
        >
          <PanelLeftOpen className="h-4 w-4" />
        </button>
      </aside>
    );
  }

  return (
    <aside className="w-64 shrink-0 border-r flex flex-col bg-muted/20">
      <div className="flex items-center justify-between p-3">
        <Button size="sm" variant="default" className="flex-1 mr-2">
          <Link href="/app" className="flex items-center gap-1.5">
            <Plus className="h-4 w-4" /> New chat
          </Link>
        </Button>
        <button
          aria-label="Collapse sidebar"
          className="text-muted-foreground hover:text-foreground p-1"
          onClick={() => setCollapsed(true)}
        >
          <PanelLeftClose className="h-4 w-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 pb-4 space-y-4 text-sm">
        {loading ? (
          <div className="flex items-center gap-2 text-muted-foreground p-3">
            <Loader2 className="h-3 w-3 animate-spin" /> Loading…
          </div>
        ) : null}

        {/* Datasets */}
        {datasets.length > 0 ? (
          <div>
            <p className="px-2 py-1 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              Datasets
            </p>
            <ul>
              {datasets.map((d) => {
                const active = activeDatasetId === d.id;
                const confirming = confirmDelete === d.id;
                return (
                  <li key={d.id} className="group/item relative">
                    <Link
                      href={`/app/datasets/${d.id}`}
                      onClick={() => setConfirmDelete(null)}
                      className={`flex items-center gap-2 px-2 py-1.5 rounded-md transition-colors pr-7 ${
                        active
                          ? "bg-primary/10 text-foreground"
                          : "text-muted-foreground hover:bg-muted hover:text-foreground"
                      }`}
                    >
                      <Database className="h-3.5 w-3.5 shrink-0" />
                      <span className="truncate flex-1">{d.name}</span>
                      <span className="text-xs text-muted-foreground/70 shrink-0">
                        {d.rowCount.toLocaleString()}
                      </span>
                    </Link>
                    <button
                      onClick={(e) => { e.preventDefault(); void handleDeleteDataset(d.id); }}
                      className={`absolute right-1 top-1/2 -translate-y-1/2 p-1 rounded transition-colors ${
                        confirming
                          ? "text-destructive"
                          : "text-muted-foreground/0 group-hover/item:text-muted-foreground hover:!text-destructive"
                      }`}
                      title={confirming ? "Click again to confirm" : "Delete dataset"}
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        ) : null}

        {/* Chats */}
        {chats.length > 0 ? (
          <div>
            <p className="px-2 py-1 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              Recent chats
            </p>
            <ul>
              {chats.map((c) => {
                const active = activeChatId === c.id;
                const confirming = confirmDelete === c.id;
                const label =
                  c.title && c.title !== "New chat"
                    ? c.title
                    : c.datasetName || "Chat";
                return (
                  <li key={c.id} className="group/item relative">
                    <Link
                      href={`/app/chat/${c.id}`}
                      onClick={() => setConfirmDelete(null)}
                      className={`flex items-start gap-2 px-2 py-1.5 rounded-md transition-colors pr-7 ${
                        active
                          ? "bg-primary/10 text-foreground"
                          : "text-muted-foreground hover:bg-muted hover:text-foreground"
                      }`}
                    >
                      <MessageSquare className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-foreground/90">{label}</p>
                        <p className="text-xs text-muted-foreground/70 truncate">
                          {c.datasetName} · {timeAgo(c.lastMessageAt ?? c.updatedAt)}
                        </p>
                      </div>
                    </Link>
                    <button
                      onClick={(e) => { e.preventDefault(); void handleDeleteChat(c.id); }}
                      className={`absolute right-1 top-1/2 -translate-y-1/2 p-1 rounded transition-colors ${
                        confirming
                          ? "text-destructive"
                          : "text-muted-foreground/0 group-hover/item:text-muted-foreground hover:!text-destructive"
                      }`}
                      title={confirming ? "Click again to confirm" : "Delete chat"}
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        ) : null}

        {!loading && datasets.length === 0 && chats.length === 0 ? (
          <div className="text-muted-foreground text-xs p-3">
            No data yet. Upload a CSV to get started.
          </div>
        ) : null}
      </div>

      {/* Settings link pinned to bottom */}
      <div className="border-t p-2">
        <Link
          href="/app/settings"
          className={`flex items-center gap-2 px-2 py-1.5 rounded-md text-sm transition-colors ${
            pathname === "/app/settings"
              ? "bg-primary/10 text-foreground"
              : "text-muted-foreground hover:bg-muted hover:text-foreground"
          }`}
        >
          <Settings className="h-4 w-4 shrink-0" />
          Settings
        </Link>
      </div>
    </aside>
  );
}
