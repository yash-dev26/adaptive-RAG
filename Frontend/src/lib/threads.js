import { getSessionId } from "./session.js";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api/v1";

function sessionHeaders() {
  return { "X-Session-Id": getSessionId() };
}

export async function fetchThreadList() {
  const res = await fetch(`${API_BASE}/threads/`, { headers: sessionHeaders() });
  if (!res.ok) throw new Error(`Failed to load threads (${res.status})`);
  const data = await res.json();
  return data.threads;
}

export async function fetchThreadMessages(threadId) {
  const res = await fetch(`${API_BASE}/threads/${encodeURIComponent(threadId)}/messages`, {
    headers: sessionHeaders(),
  });
  if (!res.ok) throw new Error(`Failed to load thread (${res.status})`);
  return res.json();
}