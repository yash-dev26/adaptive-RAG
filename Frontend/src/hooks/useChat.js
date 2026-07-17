import { useEffect, useState, useCallback } from "react";
import { randomId } from "../lib/utils.js";
import { useApiKeys } from "../context/ApiKeysContext.jsx";
import { useApiHeaders } from "./useApiHeaders.js";
import { useEntries } from "./useEntries.js";
import { useThreads } from "./useThreads.js";
import { useFileUpload } from "./useFileUpload.js";
import { useChatStream } from "./useChatStream.js";

export function useChat() {
  const { hasOpenAiKey, openModal } = useApiKeys();
  const { buildHeaders } = useApiHeaders();
  const { entries, setEntries, addEntry, resetEntries } = useEntries();
  const [threadId, setThreadId] = useState(() => `t_${randomId()}`);

  const {
    threads,
    isLoadingThreads,
    isLoadingThread,
    refreshThreads,
    loadThreadMessages,
  } = useThreads();

  const {
    uploadedFileName,
    hasUploadedFile,
    isUploading,
    ingestedFileId,
    uploadFile,
    resetUpload,
    applyThreadFile,
  } = useFileUpload({ buildHeaders, addEntry, hasOpenAiKey, openModal });

  const { isLoading, streamTrace, streamMessage, submitQuery, resetStream } = useChatStream({
    buildHeaders,
    addEntry,
    threadId,
    setThreadId,
    hasUploadedFile,
    ingestedFileId,
    uploadedFileName,
    hasOpenAiKey,
    openModal,
    onSettled: refreshThreads,
  });

  useEffect(() => {
    refreshThreads();
  }, [refreshThreads]);

  const newThread = useCallback(() => {
    resetEntries();
    setThreadId(`t_${randomId()}`);
    resetStream();
    resetUpload();
  }, [resetEntries, resetStream, resetUpload]);

  const loadThread = useCallback(
    async (thread) => {
      if (isLoading || thread.thread_id === threadId) return;

      try {
        const pairedEntries = await loadThreadMessages(thread.thread_id);
        setEntries(pairedEntries);
        setThreadId(thread.thread_id);
        resetStream();

        // Older threads (created before file_name was persisted) won't have
        // it — fall back to the id-based label rather than showing nothing.
        applyThreadFile(thread);
      } catch {
        addEntry({
          query: "Load conversation",
          answer: "Could not load this conversation. Try again in a moment.",
          sources: [],
          confidence: 0,
          pipeline: [{ name: "threads", status: "error", detail: "Failed to load thread.", badge: null }],
          cacheHit: false,
        });
      }
    },
    [isLoading, threadId, loadThreadMessages, setEntries, resetStream, applyThreadFile, addEntry]
  );

  return {
    entries,
    isLoading,
    streamTrace,
    streamMessage,
    threadId,
    uploadedFileName,
    hasUploadedFile,
    isUploading,
    ingestedFileId,
    submitQuery,
    uploadFile,
    newThread,
    threads,
    isLoadingThreads,
    isLoadingThread,
    loadThread,
  };
}