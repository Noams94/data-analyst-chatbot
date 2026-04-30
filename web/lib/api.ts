/**
 * Tiny client for the FastAPI backend. The base URL is read from
 * NEXT_PUBLIC_API_URL (defaults to http://localhost:8001 in dev).
 */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

export interface DatasetSummary {
  id: string;
  name: string;
  columns: string[];
  rowCount: number;
  sizeBytes: number;
  createdAt: string;
}

export interface ChartPart {
  id: string;
  url: string;
  title: string;
  chartType: string;
}

export interface SnippetPart {
  id: string;
  type: "analysis" | "sql" | "chart";
  code: string;
}

export interface MessageDTO {
  id: string;
  chatId: string;
  role: "user" | "assistant" | "tool";
  content: string;
  pinned: boolean;
  charts: ChartPart[];
  snippets: SnippetPart[];
  createdAt: string;
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

export interface ChatDTO {
  id: string;
  datasetId: string;
  title: string;
  messages: MessageDTO[];
}

export const api = {
  baseUrl: BASE,

  async uploadDataset(file: File): Promise<DatasetSummary> {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${BASE}/datasets`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async getDataset(datasetId: string): Promise<DatasetSummary> {
    const res = await fetch(`${BASE}/datasets/${datasetId}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async createChat(datasetId: string): Promise<{ id: string; datasetId: string; title: string }> {
    const res = await fetch(`${BASE}/chats`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: datasetId }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async getChat(chatId: string): Promise<ChatDTO> {
    const res = await fetch(`${BASE}/chats/${chatId}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async setMessagePinned(messageId: string, pinned: boolean): Promise<void> {
    const res = await fetch(`${BASE}/messages/${messageId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pinned }),
    });
    if (!res.ok) throw new Error(await res.text());
  },

  async getReport(chatId: string): Promise<ReportDoc> {
    const res = await fetch(`${BASE}/chats/${chatId}/report`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  /** Returns the absolute URL for a chart PNG. Use directly in <img src>. */
  chartUrl(idOrPath: string): string {
    return idOrPath.startsWith("http") ? idOrPath : `${BASE}${idOrPath.startsWith("/") ? "" : "/"}${idOrPath}`;
  },

  /** Build the SSE endpoint URL — caller is responsible for opening the stream. */
  messagesEndpoint(chatId: string): string {
    return `${BASE}/chats/${chatId}/messages`;
  },
};
