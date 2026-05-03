/**
 * Tiny client for the FastAPI backend.
 *
 * All non-public endpoints expect a Clerk session JWT in the
 * `Authorization: Bearer <token>` header. The hook `useApi()` (below) wires
 * this up automatically using Clerk's `useAuth().getToken()`. Components
 * should call `const api = useApi()` rather than importing the bare module.
 */

import { useAuth } from "@clerk/nextjs";
import { useMemo } from "react";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

export interface DatasetSummary {
  id: string;
  name: string;
  columns: string[];
  rowCount: number;
  sizeBytes: number;
  createdAt: string;
}

export interface ChatSummary {
  id: string;
  datasetId: string;
  datasetName: string;
  title: string;
  messageCount: number;
  lastMessageAt: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface ColumnInfo {
  name: string;
  dtype: string;
  nullCount: number;
  nullPct: number;
  uniqueCount: number;
  sampleValues: string[];
}

export interface DatasetPreview {
  id: string;
  name: string;
  rowCount: number;
  createdAt: string;
  columns: ColumnInfo[];
  numericSummary: Record<string, Record<string, number | null>>;
  sampleRows: Record<string, unknown>[];
}

export interface ChartPart {
  id: string;
  /** `data:image/png;base64,...` — usable directly in <img src>. */
  dataUrl: string;
  title: string;
  chartType: string;
}

export interface SnippetPart {
  id: string;
  type: "analysis" | "sql" | "chart";
  code: string;
}

export interface PlotlyChartPart {
  id: string;
  spec: string;
  title: string;
}

export interface DashboardChartPart {
  id: string;
  chatId: string;
  spec: string;
  title: string;
  position: number;
}

export interface MessageDTO {
  id: string;
  chatId: string;
  role: "user" | "assistant" | "tool";
  content: string;
  pinned: boolean;
  charts: ChartPart[];
  snippets: SnippetPart[];
  plotlyCharts: PlotlyChartPart[];
  createdAt: string;
}

export interface ChatDTO {
  id: string;
  datasetId: string;
  title: string;
  messages: MessageDTO[];
}

export interface UserSettings {
  provider: "anthropic" | "ollama";
  anthropicModel: string;
  anthropicApiKey: string;   // masked display value
  hasAnthropicKey: boolean;
  ollamaModel: string;
  ollamaBaseUrl: string;
}

export interface ReportSection {
  id: string;
  question: string;
  bodyMd: string;
  charts: { id: string; title: string; chartType: string; dataUrl: string }[];
  snippets: SnippetPart[];
  createdAt: string;
}

export interface ReportDoc {
  chatId: string;
  datasetName: string;
  datasetMeta: { rowCount: number; columns: string[] };
  generatedAt: string;
  sections: ReportSection[];
}

type GetToken = () => Promise<string | null>;

async function authedFetch(
  getToken: GetToken,
  url: string,
  init?: RequestInit,
): Promise<Response> {
  const token = await getToken();
  const headers = new Headers(init?.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return fetch(url, { ...init, headers });
}

export function makeApi(getToken: GetToken) {
  return {
    baseUrl: BASE,

    async uploadDataset(file: File): Promise<DatasetSummary> {
      const fd = new FormData();
      fd.append("file", file);
      const res = await authedFetch(getToken, `${BASE}/datasets`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async getDataset(datasetId: string): Promise<DatasetSummary> {
      const res = await authedFetch(getToken, `${BASE}/datasets/${datasetId}`);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async deleteDataset(datasetId: string): Promise<void> {
      const res = await authedFetch(getToken, `${BASE}/datasets/${datasetId}`, { method: "DELETE" });
      if (!res.ok) throw new Error(await res.text());
    },

    async deleteChat(chatId: string): Promise<void> {
      const res = await authedFetch(getToken, `${BASE}/chats/${chatId}`, { method: "DELETE" });
      if (!res.ok) throw new Error(await res.text());
    },

    async listDatasets(): Promise<DatasetSummary[]> {
      const res = await authedFetch(getToken, `${BASE}/datasets`);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async getDatasetPreview(datasetId: string): Promise<DatasetPreview> {
      const res = await authedFetch(getToken, `${BASE}/datasets/${datasetId}/preview`);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async listChats(datasetId?: string): Promise<ChatSummary[]> {
      const url = datasetId
        ? `${BASE}/chats?dataset_id=${encodeURIComponent(datasetId)}`
        : `${BASE}/chats`;
      const res = await authedFetch(getToken, url);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async createChat(datasetId: string): Promise<{ id: string; datasetId: string; title: string }> {
      const res = await authedFetch(getToken, `${BASE}/chats`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: datasetId }),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async getChat(chatId: string): Promise<ChatDTO> {
      const res = await authedFetch(getToken, `${BASE}/chats/${chatId}`);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async setMessagePinned(messageId: string, pinned: boolean): Promise<void> {
      const res = await authedFetch(getToken, `${BASE}/messages/${messageId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pinned }),
      });
      if (!res.ok) throw new Error(await res.text());
    },

    async getSettings(): Promise<UserSettings> {
      const res = await authedFetch(getToken, `${BASE}/settings`);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async patchSettings(updates: Partial<{
      provider: string;
      anthropicModel: string;
      anthropicApiKey: string;
      ollamaModel: string;
      ollamaBaseUrl: string;
    }>): Promise<UserSettings> {
      const res = await authedFetch(getToken, `${BASE}/settings`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async getReport(chatId: string): Promise<ReportDoc> {
      const res = await authedFetch(getToken, `${BASE}/chats/${chatId}/report`);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async getDashboard(chatId: string): Promise<DashboardChartPart[]> {
      const res = await authedFetch(getToken, `${BASE}/chats/${chatId}/dashboard`);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async clearDashboard(chatId: string): Promise<void> {
      const res = await authedFetch(getToken, `${BASE}/chats/${chatId}/dashboard`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error(await res.text());
    },

    /**
     * Build the SSE endpoint URL. SSE streams need the token on the request
     * we open with `fetch()`; useChatStream handles that itself.
     */
    messagesEndpoint(chatId: string): string {
      return `${BASE}/chats/${chatId}/messages`;
    },

    /**
     * Get an auth header for the SSE call. Returns an empty record when
     * there's no token (anonymous dev mode).
     */
    async authHeader(): Promise<Record<string, string>> {
      const token = await getToken();
      return token ? { Authorization: `Bearer ${token}` } : {};
    },

  };
}

export type Api = ReturnType<typeof makeApi>;

/** React hook — use this from components. */
export function useApi(): Api {
  const { getToken } = useAuth();
  return useMemo(() => makeApi(() => getToken()), [getToken]);
}
