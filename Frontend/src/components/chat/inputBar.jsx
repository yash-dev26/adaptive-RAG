import { useState } from "react";
import { Button, Input, Separator } from "../ui";

function InputBar({ onSubmit, onFileChange, isLoading, isUploading, fileInputRef }) {
  const [value, setValue] = useState("");

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function submit() {
    if (!value.trim() || isLoading) return;
    onSubmit(value.trim());
    setValue("");
  }

  return (
    <div className="border-t border-zinc-800 bg-zinc-950/60 px-6 py-4">
      <div className="flex items-center gap-3 max-w-3xl">
        <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.txt,.docx"
        className="hidden"
        onChange={(e) => {
            onFileChange(e.target.files?.[0]);
            e.target.value = "";
        }}
        />
        <Button
          variant="ghost"
          size="icon"
          disabled={isUploading}
          onClick={() => fileInputRef.current?.click()}
          title="Upload document"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.44 11.05l-8.49 8.49a5 5 0 01-7.07-7.07l8.49-8.49a3.5 3.5 0 114.95 4.95l-8.5 8.49a2 2 0 01-2.83-2.83l7.78-7.78" />
          </svg>
        </Button>

        <Separator orientation="vertical" className="h-5" />

        <Input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask something about your document…"
          disabled={isLoading}
          className="flex-1"
        />

        <Button
          variant="accent"
          size="md"
          onClick={submit}
          disabled={isLoading || isUploading || !value.trim()}
          className="font-mono shrink-0"
        >
          {isLoading ? "···" : "Send"}
        </Button>
      </div>
    </div>
  );
}

export default InputBar;