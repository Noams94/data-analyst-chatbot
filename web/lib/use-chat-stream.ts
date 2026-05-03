"use client";

/**
 * SSE consumer for the FastAPI /chats/:id/messages endpoint.
 *
 * The backend emits these events:
 *   token       — text delta
 *   tool_start  — {name, args}
 *   tool_end    — {name}
 *   chart       — {id, url, title, chartType}
 *   snippet     — {id, type, code}
 *   error       — {message}
 *   done        — {messageId}
 */

import { useCallback, useReducer, useRef } from "react";
import { useApi, type ChartPart, type DashboardChartPart, type MessageDTO, type PlotlyChartPart, type SnippetPart } from "./api";

export type StreamingMessage = {
  id: string;          // local placeholder until `done` brings the server id
  role: "assistant";
  content: string;
  charts: ChartPart[];
  snippets: SnippetPart[];
  plotlyCharts: PlotlyChartPart[];
  toolActive: string | null;
  isStreaming: boolean;
  error: string | null;
};

type State = {
  messages: MessageDTO[];                  // fully persisted turns from the server
  streaming: StreamingMessage | null;      // the in-flight assistant message
  suggestions: string[];                   // follow-up question chips
};

type Action =
  | { type: "reset"; messages: MessageDTO[] }
  | { type: "user_sent"; content: string; tempId: string }
  | { type: "stream_start"; tempId: string }
  | { type: "token"; text: string }
  | { type: "tool_start"; name: string }
  | { type: "tool_end" }
  | { type: "chart"; chart: ChartPart }
  | { type: "plotly_chart"; chart: PlotlyChartPart }
  | { type: "snippet"; snippet: SnippetPart }
  | { type: "error"; message: string }
  | { type: "done"; messageId: string }
  | { type: "set_pinned"; messageId: string; pinned: boolean }
  | { type: "suggestions"; items: string[] }
  | { type: "dashboard_chart"; chart: DashboardChartPart };

function blankStreaming(tempId: string): StreamingMessage {
  return {
    id: tempId,
    role: "assistant",
    content: "",
    charts: [],
    snippets: [],
    plotlyCharts: [],
    toolActive: null,
    isStreaming: true,
    error: null,
  };
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "reset":
      return { messages: action.messages, streaming: null, suggestions: [] };
    case "user_sent":
      return {
        messages: [
          ...state.messages,
          {
            id: action.tempId,
            chatId: "",
            role: "user",
            content: action.content,
            pinned: true,
            charts: [],
            snippets: [],
            plotlyCharts: [],
            createdAt: new Date().toISOString(),
          },
        ],
        streaming: null,
        suggestions: [],
      };
    case "stream_start":
      return { ...state, streaming: blankStreaming(action.tempId) };
    case "token":
      if (!state.streaming) return state;
      return { ...state, streaming: { ...state.streaming, content: state.streaming.content + action.text } };
    case "tool_start":
      if (!state.streaming) return state;
      return { ...state, streaming: { ...state.streaming, toolActive: action.name } };
    case "tool_end":
      if (!state.streaming) return state;
      return { ...state, streaming: { ...state.streaming, toolActive: null } };
    case "chart":
      if (!state.streaming) return state;
      return { ...state, streaming: { ...state.streaming, charts: [...state.streaming.charts, action.chart] } };
    case "plotly_chart":
      if (!state.streaming) return state;
      return { ...state, streaming: { ...state.streaming, plotlyCharts: [...state.streaming.plotlyCharts, action.chart] } };
    case "snippet":
      if (!state.streaming) return state;
      return { ...state, streaming: { ...state.streaming, snippets: [...state.streaming.snippets, action.snippet] } };
    case "error":
      if (!state.streaming) return state;
      return { ...state, streaming: { ...state.streaming, error: action.message } };
    case "done":
      if (!state.streaming) return state;
      // Promote the streaming buffer into a persisted message.
      const finalized: MessageDTO = {
        id: action.messageId,
        chatId: "",
        role: "assistant",
        content: state.streaming.content,
        pinned: true,
        charts: state.streaming.charts,
        snippets: state.streaming.snippets,
        plotlyCharts: state.streaming.plotlyCharts,
        createdAt: new Date().toISOString(),
      };
      return { messages: [...state.messages, finalized], streaming: null, suggestions: state.suggestions };
    case "set_pinned":
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.messageId ? { ...m, pinned: action.pinned } : m,
        ),
      };
    case "suggestions":
      return { ...state, suggestions: action.items };
    case "dashboard_chart":
      window.dispatchEvent(
        new CustomEvent("dashboard-chart-added", { detail: action.chart }),
      );
      return state;
  }
}

function parseSseChunk(buffer: string): { events: { event: string; data: string }[]; remainder: string } {
  // SSE frames are separated by a blank line. Split on \n\n, keep the trailing
  // partial frame in remainder.
  const parts = buffer.split("\n\n");
  const remainder = parts.pop() ?? "";
  const events = parts
    .map((frame) => {
      const lines = frame.split("\n");
      let event = "message";
      let data = "";
      for (const line of lines) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      return { event, data };
    })
    .filter((e) => e.data || e.event !== "message");
  return { events, remainder };
}

export function useChatStream(initialMessages: MessageDTO[] = []) {
  const api = useApi();
  const [state, dispatch] = useReducer(reducer, { messages: initialMessages, streaming: null, suggestions: [] });
  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback((messages: MessageDTO[]) => {
    dispatch({ type: "reset", messages });
  }, []);

  const send = useCallback(async (chatId: string, content: string) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const tempUserId = `local-user-${Date.now()}`;
    const tempStreamId = `local-stream-${Date.now()}`;
    dispatch({ type: "user_sent", content, tempId: tempUserId });
    dispatch({ type: "stream_start", tempId: tempStreamId });

    try {
      const auth = await api.authHeader();
      const res = await fetch(api.messagesEndpoint(chatId), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
          ...auth,
        },
        body: JSON.stringify({ content }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => res.statusText);
        dispatch({ type: "error", message: text || `HTTP ${res.status}` });
        dispatch({ type: "done", messageId: tempStreamId });
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const { events, remainder } = parseSseChunk(buffer);
        buffer = remainder;
        for (const ev of events) {
          let parsed: unknown = {};
          try {
            parsed = JSON.parse(ev.data);
          } catch {
            // ignore malformed frame
            continue;
          }
          const data = parsed as Record<string, unknown>;
          switch (ev.event) {
            case "token":
              dispatch({ type: "token", text: String(data.text ?? "") });
              break;
            case "tool_start":
              dispatch({ type: "tool_start", name: String(data.name ?? "") });
              break;
            case "tool_end":
              dispatch({ type: "tool_end" });
              break;
            case "chart":
              dispatch({ type: "chart", chart: data as unknown as ChartPart });
              break;
            case "plotly_chart":
              dispatch({ type: "plotly_chart", chart: data as unknown as PlotlyChartPart });
              break;
            case "snippet":
              dispatch({ type: "snippet", snippet: data as unknown as SnippetPart });
              break;
            case "error":
              dispatch({ type: "error", message: String(data.message ?? "stream error") });
              break;
            case "done":
              dispatch({ type: "done", messageId: String(data.messageId ?? tempStreamId) });
              break;
            case "title":
              if (chatId && typeof data.title === "string") {
                window.dispatchEvent(new CustomEvent("chat-title-changed", {
                  detail: { chatId, title: data.title },
                }));
              }
              break;
            case "suggestions":
              if (Array.isArray(data.items)) {
                dispatch({ type: "suggestions", items: data.items as string[] });
              }
              break;
            case "dashboard_chart":
              dispatch({ type: "dashboard_chart", chart: data as unknown as DashboardChartPart });
              break;
          }
        }
      }
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      dispatch({ type: "error", message: (e as Error).message });
      dispatch({ type: "done", messageId: tempStreamId });
    }
  }, [api]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const setPinned = useCallback(async (messageId: string, pinned: boolean) => {
    // Optimistic; revert on failure.
    dispatch({ type: "set_pinned", messageId, pinned });
    try {
      await api.setMessagePinned(messageId, pinned);
    } catch {
      dispatch({ type: "set_pinned", messageId, pinned: !pinned });
    }
  }, [api]);

  return { state, send, reset, stop, setPinned };
}
