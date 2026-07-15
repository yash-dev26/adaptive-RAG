import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { useChat } from "../hooks/useChat";
import { Button, Badge, Card, CardHeader, CardFooter, Collapsible, Input, Separator, ScrollArea } from "../components/ui";
import SessionSidebar from "../components/chat/SessionSidebar";
import ContextPanel from "../components/chat/DocPanel";
import ResponseEntry from "../components/chat/ResponseEntry";
import LoadingIndicator from "../components/chat/LoadingEntry";
import InputBar from "../components/chat/inputBar";
import EmptyState from "../components/chat/emptyState";

export default function ChatPage() {
  const {
  entries, isLoading, streamTrace, streamMessage,
  threadId, uploadedFileName, hasUploadedFile, isUploading, ingestedFileId,
  submitQuery, uploadFile, newThread,
  threads, isLoadingThreads, loadThread,
  } = useChat();

  const feedRef = useRef(null);
  const fileInputRef = useRef(null);
  const [pendingQuery, setPendingQuery] = useState("");

  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [entries, isLoading]);

  // Handle suggestion selection
  useEffect(() => {
    if (pendingQuery) {
      submitQuery(pendingQuery);
      setPendingQuery("");
    }
  }, [pendingQuery]);

  const latestSources = entries.length > 0 ? entries[entries.length - 1].sources : [];

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100 overflow-hidden">
    <SessionSidebar
        threadId={threadId}
        queryCount={entries.filter((e) => !e.query.startsWith("Uploaded:")).length}
        uploadedFileName={uploadedFileName}
        ingestedFileId={ingestedFileId}
        isUploading={isUploading}
        hasUploadedFile={hasUploadedFile}
        onNewThread={newThread}
        threads={threads}
        isLoadingThreads={isLoadingThreads}
        onSelectThread={loadThread}
    />

      <div className="flex-1 flex flex-col overflow-hidden">
        <div
          ref={feedRef}
          className="flex-1 overflow-y-auto px-8 py-6 flex flex-col gap-5"
        >
          {entries.length === 0 && !isLoading ? (
            <EmptyState onSelect={setPendingQuery} />
          ) : (
            entries.map((entry, i) => <ResponseEntry key={i} entry={entry} />)
          )}
          {isLoading && (
            <LoadingIndicator message={streamMessage} trace={streamTrace} />
          )}
        </div>

        <InputBar
          onSubmit={submitQuery}
          onFileChange={uploadFile}
          isLoading={isLoading}
          isUploading={isUploading}
          fileInputRef={fileInputRef}
        />
      </div>

      <ContextPanel sources={latestSources} />

      <style>{`
        @keyframes pulse-dot {
          0%, 80%, 100% { opacity: 0.2; transform: translateY(0); }
          40% { opacity: 1; transform: translateY(-2px); }
        }
      `}</style>
    </div>
  );
}