import { Button, Separator, ScrollArea } from "../ui";
import { formatRelativeTime } from "../../lib/utils.js";

function SessionSidebar({
  threadId,
  queryCount,
  uploadedFileName,
  ingestedFileId,
  isUploading,
  hasUploadedFile,
  onNewThread,
  threads = [],
  isLoadingThreads,
  onSelectThread,
}) {
  return (
    <aside className="w-60 shrink-0 border-r border-zinc-800 flex flex-col p-4 gap-4">
      <div>
        <p className="text-[10px] font-mono text-zinc-700 uppercase tracking-widest mb-2">Session</p>
        <Button variant="outline" size="sm" className="w-full justify-start font-mono" onClick={onNewThread}>
          + New thread
        </Button>
      </div>

      <Separator />

      <div className="flex-1 min-h-0 flex flex-col">
        <p className="text-[10px] font-mono text-zinc-700 uppercase tracking-widest mb-2">Recent chats</p>
        <ScrollArea className="flex-1 -mx-1 px-1">
          {isLoadingThreads && threads.length === 0 ? (
            <p className="text-[11px] text-zinc-600 px-1">Loading…</p>
          ) : threads.length === 0 ? (
            <p className="text-[11px] text-zinc-600 px-1">No conversations yet.</p>
          ) : (
            <div className="space-y-1">
              {threads.map((t) => {
                const isActive = t.thread_id === threadId;
                return (
                  <button
                    key={t.thread_id}
                    onClick={() => onSelectThread?.(t)}
                    className={[
                      "w-full text-left rounded-md px-2 py-1.5 transition-colors",
                      isActive ? "bg-zinc-800 text-zinc-100" : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200",
                    ].join(" ")}
                  >
                    <p className="text-[11px] truncate">{t.title}</p>
                    <p className="text-[10px] font-mono text-zinc-600">{formatRelativeTime(t.updated_at)}</p>
                  </button>
                );
              })}
            </div>
          )}
        </ScrollArea>
      </div>

      <Separator />

      <div className="space-y-1">
        <p className="text-[10px] font-mono text-zinc-600 uppercase tracking-widest mb-2">Thread</p>
        {[
          ["Queries", queryCount],
        ].map(([label, value]) => (
          <div key={label} className="flex justify-between items-center">
            <span className="text-[11px] text-zinc-600">{label}</span>
            <span className="font-mono text-[11px] text-zinc-400 truncate max-w-28">{value}</span>
          </div>
        ))}
      </div>

      <Separator />

      <div className="space-y-1">
        <p className="text-[10px] font-mono text-zinc-700 uppercase tracking-widest mb-2">Document</p>
        <div className="rounded-md border border-zinc-800 bg-zinc-900 p-3 space-y-1">
          <p className="text-[11px] text-zinc-300 truncate">{uploadedFileName || "None"}</p>
          <div className="flex items-center gap-1.5">
            <span
              className={[
                "w-1.5 h-1.5 rounded-full",
                isUploading ? "bg-amber-400" : hasUploadedFile ? "bg-emerald-400" : "bg-zinc-700",
              ].join(" ")}
            />
            <span className="font-mono text-[10px] text-zinc-500">
              {isUploading ? "Uploading…" : hasUploadedFile ? `Ready · ${ingestedFileId?.slice(0, 8)}` : "No file"}
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
}

export default SessionSidebar;