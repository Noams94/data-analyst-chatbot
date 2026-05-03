export type Role = "user" | "assistant" | "tool";

export type SnippetType = "analysis" | "sql" | "chart";

export interface Dataset {
  id: string;
  name: string;
  columns: string[];
  rowCount: number;
  sizeBytes: number;
  createdAt: string;
}

export interface Chart {
  id: string;
  url: string;
  title: string;
  chartType: string;
}

export interface CodeSnippet {
  id: string;
  type: SnippetType;
  code: string;
}

export interface Message {
  id: string;
  chatId: string;
  role: Role;
  content: string;
  charts: Chart[];
  snippets: CodeSnippet[];
  createdAt: string;
}

export interface Chat {
  id: string;
  datasetId: string;
  title: string;
  createdAt: string;
  updatedAt: string;
}

// SSE event payloads emitted by the FastAPI service.
export type StreamEvent =
  | { event: "token"; data: { text: string } }
  | { event: "tool_start"; data: { name: string; args: Record<string, unknown> } }
  | { event: "tool_end"; data: { name: string } }
  | { event: "chart"; data: Chart }
  | { event: "snippet"; data: CodeSnippet }
  | { event: "done"; data: { messageId: string } }
  | { event: "error"; data: { message: string } };
