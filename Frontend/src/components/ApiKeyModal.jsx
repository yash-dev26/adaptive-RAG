import { useEffect, useState } from "react";
import { Button, Input, Card, CardHeader, CardContent, CardFooter, Badge } from "./ui";
import { useApiKeys } from "../context/ApiKeysContext.jsx";

export default function ApiKeyModal() {
  const { isModalOpen, closeModal, openaiKey, groqKey, saveKeys, clearKeys } = useApiKeys();

  const [openaiInput, setOpenaiInput] = useState(openaiKey);
  const [groqInput, setGroqInput] = useState(groqKey);
  const [error, setError] = useState("");

  useEffect(() => {
    if (isModalOpen) {
      setOpenaiInput(openaiKey);
      setGroqInput(groqKey);
      setError("");
    }
  }, [isModalOpen, openaiKey, groqKey]);

  if (!isModalOpen) return null;

  function handleSave() {
    if (!openaiInput.trim()) {
      setError("An OpenAI API key is required — it powers embeddings and answer generation.");
      return;
    }
    saveKeys({ openaiKey: openaiInput.trim(), groqKey: groqInput.trim() });
    closeModal();
  }

  function handleForget() {
    clearKeys();
    setOpenaiInput("");
    setGroqInput("");
    setError("");
  }

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm px-4">
      <Card className="w-full max-w-md">
        <CardHeader className="border-b border-zinc-800">
          <h2 className="text-lg font-semibold text-zinc-100">Bring your own API keys</h2>
          <p className="text-sm text-zinc-500 mt-1">
            Keys are stored only in this browser's local storage and sent directly to your own
            backend as request headers. They're never logged or persisted server-side.
          </p>
        </CardHeader>

        <CardContent className="space-y-4">
          <div>
            <label className="flex items-center gap-2 text-xs font-mono uppercase tracking-wide text-zinc-400 mb-2">
              OpenAI API key <Badge variant="accent">required</Badge>
            </label>
            <Input
              type="password"
              placeholder="sk-..."
              value={openaiInput}
              onChange={(e) => setOpenaiInput(e.target.value)}
              autoComplete="off"
            />
            <p className="text-xs text-zinc-600 mt-1">Used for embeddings and final answer generation.</p>
          </div>

          <div>
            <label className="flex items-center gap-2 text-xs font-mono uppercase tracking-wide text-zinc-400 mb-2">
              Groq API key <Badge>optional</Badge>
            </label>
            <Input
              type="password"
              placeholder="gsk_... (optional)"
              value={groqInput}
              onChange={(e) => setGroqInput(e.target.value)}
              autoComplete="off"
            />
            <p className="text-xs text-zinc-600 mt-1">
              Speeds up query rewriting and retrieval evaluation. Without it, those steps run on
              OpenAI instead.
            </p>
          </div>

          {error && <p className="text-sm text-red-400">{error}</p>}
        </CardContent>

        <CardFooter className="flex justify-between gap-2">
          <Button
            variant="ghost"
            size="md"
            onClick={handleForget}
            disabled={!openaiKey && !groqKey}
            className="text-red-400 hover:text-red-300"
          >
            Forget keys
          </Button>
          <div className="flex gap-2">
            <Button variant="ghost" size="md" onClick={closeModal}>
              Cancel
            </Button>
            <Button variant="accent" size="md" onClick={handleSave}>
              Save keys
            </Button>
          </div>
        </CardFooter>
      </Card>
    </div>
  );
}