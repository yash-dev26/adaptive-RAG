import { useEffect } from "react";
import { Button } from "./ui";
import { useApiKeys } from "../context/ApiKeysContext.jsx";

export default function RequireApiKey({ children }) {
  const { hasOpenAiKey, openModal } = useApiKeys();

  useEffect(() => {
    if (!hasOpenAiKey) {
      openModal();
    }
  }, [hasOpenAiKey, openModal]);

  if (!hasOpenAiKey) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-zinc-950 text-zinc-100 px-4">
        <div className="text-center max-w-sm space-y-4">
          <p className="text-lg font-semibold">API key required</p>
          <p className="text-sm text-zinc-500">
            This is a bring-your-own-key app — add an OpenAI API key to unlock the chat.
            Groq and Tavily are optional. Your keys stay in this browser and are never stored on
            our server.
          </p>
          <Button variant="accent" size="md" className="font-mono" onClick={openModal}>
            Add API key
          </Button>
        </div>
      </div>
    );
  }

  return children;
}