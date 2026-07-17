import { useRef, useState, useCallback } from "react";
import { parseSseBlock, upsertTrace, mapSources } from "../lib/utils.js";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api/v1";
const FIRST_REQUEST_BOOT_HINT_DELAY_MS = 8000;
const FIRST_REQUEST_BOOT_HINT = "Backend is booting up (free tier cold start). First response can take around a minute.";

export function useChatStream({
  buildHeaders,
  addEntry,
  threadId,
  setThreadId,
  hasUploadedFile,
  ingestedFileId,
  uploadedFileName,
  hasOpenAiKey,
  openModal,
  onSettled,
}) {
  const isFirstQueryRef = useRef(true);
  const [isLoading, setIsLoading] = useState(false);
  const [streamTrace, setStreamTrace] = useState([]);
  const [streamMessage, setStreamMessage] = useState("");

  const submitQuery = useCallback(
    async (query) => {
      if (!query.trim() || isLoading) return;

      if (!hasOpenAiKey) {
        openModal();
        addEntry({
          query,
          answer: "Add your OpenAI API key first (top-right → API Keys) — it's required for embeddings and generation.",
          sources: [],
          confidence: 0,
          pipeline: [{ name: "chat", status: "error", detail: "Missing OpenAI API key.", badge: null }],
          cacheHit: false,
        });
        return;
      }

      const shouldShowFirstBootHint = isFirstQueryRef.current;
      let bootHintTimer = null;
      let hasReceivedServerEvent = false;
      let collectedTrace = [];
      setIsLoading(true);
      setStreamTrace([]);
      setStreamMessage("Checking caches…");

      if (shouldShowFirstBootHint) {
        bootHintTimer = window.setTimeout(() => {
          if (!hasReceivedServerEvent) {
            setStreamMessage(FIRST_REQUEST_BOOT_HINT);
          }
        }, FIRST_REQUEST_BOOT_HINT_DELAY_MS);
      }

      function appendTrace(entry) {
        hasReceivedServerEvent = true;
        if (bootHintTimer) {
          window.clearTimeout(bootHintTimer);
          bootHintTimer = null;
        }
        collectedTrace = upsertTrace(collectedTrace, entry);
        setStreamTrace([...collectedTrace]);
        setStreamMessage(`${entry.node}: ${entry.detail}`);
      }

      try {
        const res = await fetch(`${API_BASE}/chat/stream`, {
          method: "POST",
          headers: buildHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify({
            query,
            file_id: hasUploadedFile ? ingestedFileId : null,
            file_name: hasUploadedFile ? uploadedFileName : null,
            thread_id: threadId,
          }),
        });

        if (!res.ok) {
          if (res.status === 401) {
            openModal();
            throw new Error("Invalid or missing API key. Please check your OpenAI key.");
          }
          if (res.status === 429) {
            throw new Error("Rate limit exceeded. You can make 8 requests per minute. Please wait before trying again.");
          }
          const errorData = await res.json().catch(() => ({}));
          throw new Error(errorData?.detail || `Stream failed (${res.status})`);
        }

        if (!res.body) {
          throw new Error("No response body from server");
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let finalPayload = null;

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split("\n\n");
          buffer = blocks.pop() || "";

          for (const block of blocks) {
            const payload = parseSseBlock(block);
            if (!payload) continue;

            hasReceivedServerEvent = true;
            if (bootHintTimer) {
              window.clearTimeout(bootHintTimer);
              bootHintTimer = null;
            }

            if (payload.type === "status") {
              setStreamMessage(payload.message || "Processing…");
            } else if (payload.type === "cache_hit") {
              appendTrace({ node: payload.cache || "cache", status: "done", detail: payload.message || "Cache hit." });
            } else if (payload.type === "node") {
              appendTrace({ node: payload.node, status: payload.status || "running", detail: payload.detail || "Executing…" });
            } else if (payload.type === "final") {
              finalPayload = payload;
            } else if (payload.type === "error") {
              throw new Error(payload.message || "Stream error");
            }
          }
        }

        if (!finalPayload) throw new Error("No final response from backend.");

        if (finalPayload.thread_id) setThreadId(finalPayload.thread_id);

        addEntry({
          query,
          answer: finalPayload.response,
          sources: mapSources(finalPayload.sources),
          confidence: finalPayload.confidence ?? null,
          pipeline: collectedTrace.length
            ? collectedTrace.map((t) => ({
                name: t.node,
                status: t.status,
                detail: t.detail,
                badge: t.node.includes("rewrite") ? "rewrite" : t.node === "cache" ? "cache" : null,
              }))
            : [{ name: "chat", status: "done", detail: "Response returned.", badge: finalPayload.cached ? "cache" : null }],
          cacheHit: finalPayload.cached === true || finalPayload.cached === "semantic",
        });

        onSettled?.();
      } catch (err) {
        const message = err?.response?.data?.detail || err.message || "Request failed";
        const isRateLimit = message.includes("Rate limit");
        addEntry({
          query,
          answer: `Error: ${message}`,
          sources: [],
          confidence: 0,
          pipeline: [{ name: "chat", status: "error", detail: message, badge: isRateLimit ? "rate-limit" : null }],
          cacheHit: false,
        });
      } finally {
        if (bootHintTimer) {
          window.clearTimeout(bootHintTimer);
        }
        isFirstQueryRef.current = false;
        setStreamMessage("");
        setStreamTrace([]);
        setIsLoading(false);
      }
    },
    [isLoading, hasOpenAiKey, openModal, addEntry, buildHeaders, hasUploadedFile, ingestedFileId, uploadedFileName, threadId, setThreadId, onSettled]
  );

  const resetStream = useCallback(() => {
    setStreamTrace([]);
    setStreamMessage("");
  }, []);

  return { isLoading, streamTrace, streamMessage, submitQuery, resetStream };
}