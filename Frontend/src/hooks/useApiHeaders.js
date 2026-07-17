import { useCallback } from "react";
import { getSessionId } from "../lib/session.js";
import { useApiKeys } from "../context/ApiKeysContext.jsx";

export function useApiHeaders() {
  const { openaiKey, groqKey } = useApiKeys();

  const buildHeaders = useCallback(
    (extra = {}) => {
      const headers = {
        "X-Session-Id": getSessionId(),
        "X-OpenAI-Key": openaiKey,
        ...extra,
      };
      if (groqKey) headers["X-Groq-Key"] = groqKey;
      return headers;
    },
    [openaiKey, groqKey]
  );

  return { buildHeaders };
}