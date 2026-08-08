const OPENAI_KEY_STORAGE = "adaptive_rag_openai_key";
const GROQ_KEY_STORAGE = "adaptive_rag_groq_key";

// Keys live only in this browser's localStorage. They're attached as request
// headers (X-OpenAI-Key / X-Groq-Key) on each call and never sent anywhere
// else. Clearing them (or your browser storage) forgets them for good —
// there's no server-side copy.
//
// Tavily is NOT part of BYOK: the web-search fallback runs on a server-side
// key (1000 free credits/month is plenty for a fallback path), so there's
// nothing to collect or store for it here.

export function getApiKeys() {
  return {
    openaiKey: localStorage.getItem(OPENAI_KEY_STORAGE) || "",
    groqKey: localStorage.getItem(GROQ_KEY_STORAGE) || "",
  };
}

export function setApiKeys({ openaiKey, groqKey }) {
  if (openaiKey && openaiKey.trim()) {
    localStorage.setItem(OPENAI_KEY_STORAGE, openaiKey.trim());
  } else {
    localStorage.removeItem(OPENAI_KEY_STORAGE);
  }

  if (groqKey && groqKey.trim()) {
    localStorage.setItem(GROQ_KEY_STORAGE, groqKey.trim());
  } else {
    localStorage.removeItem(GROQ_KEY_STORAGE);
  }
}

export function clearApiKeys() {
  localStorage.removeItem(OPENAI_KEY_STORAGE);
  localStorage.removeItem(GROQ_KEY_STORAGE);
}

export function hasRequiredKeys() {
  return Boolean(localStorage.getItem(OPENAI_KEY_STORAGE));
}

export function readApiKeysSnapshot() {
  return { ...getApiKeys(), hasOpenAiKey: hasRequiredKeys() };
}