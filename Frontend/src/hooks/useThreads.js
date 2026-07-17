import { useState, useCallback } from "react";
import { fetchThreadList, fetchThreadMessages } from "../lib/threads.js";
import { pairMessagesIntoEntries } from "../lib/utils.js";

export function useThreads() {
  const [threads, setThreads] = useState([]);
  const [isLoadingThreads, setIsLoadingThreads] = useState(false);
  const [isLoadingThread, setIsLoadingThread] = useState(false);

  const refreshThreads = useCallback(async () => {
    setIsLoadingThreads(true);
    try {
      setThreads(await fetchThreadList());
    } catch {
      // fail quietly because its not a critical operation
    } finally {
      setIsLoadingThreads(false);
    }
  }, []);

  const loadThreadMessages = useCallback(async (threadId) => {
    setIsLoadingThread(true);
    try {
      const { messages } = await fetchThreadMessages(threadId);
      return pairMessagesIntoEntries(messages);
    } finally {
      setIsLoadingThread(false);
    }
  }, []);

  return {
    threads,
    isLoadingThreads,
    isLoadingThread,
    refreshThreads,
    loadThreadMessages,
  };
}