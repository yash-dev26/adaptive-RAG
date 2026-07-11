const SESSION_ID_STORAGE_KEY = "adaptive_rag_session_id";

// BYOK replacement for Clerk's user.id. Not auth — just a stable id so the
// backend can keep one browser's uploads/chat history separate from another's.
export function getSessionId() {
  let id = localStorage.getItem(SESSION_ID_STORAGE_KEY);
  if (!id) {
    id = `sess_${crypto.randomUUID()}`;
    localStorage.setItem(SESSION_ID_STORAGE_KEY, id);
  }
  return id;
}
