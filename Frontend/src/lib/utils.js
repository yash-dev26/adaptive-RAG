export function parseSseBlock(block) {
  const dataLines = block
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart());

  if (!dataLines.length) return null;

  try {
    return JSON.parse(dataLines.join("\n"));
  } catch {
    return null;
  }
}

export function upsertTrace(trace, entry) {
  return [...trace.filter((t) => t.node !== entry.node), entry].slice(-6);
}

export function derivePageFromSource(s) {
  if (!s) return null;
  const payload = s.payload || {};
  if (s.page) return s.page;
  if (payload.page) return payload.page;
  if (payload.page_number) return payload.page_number;
  const text = s.text || payload.chunk || payload.text || "";
  const m = text.match(/page[: ]+(\d+)/i) || text.match(/p\.\s*(\d+)/i);
  return m ? Number(m[1]) : null;
}

export function mapSources(rawSources = []) {
  return rawSources.map((s, idx) => ({
    id: s.id ?? s.payload?.id ?? `src_${idx}`,
    text: s.text ?? s.payload?.chunk ?? s.payload?.text ?? s.content ?? "",
    score: s.score ?? s.similarity ?? s.payload?.score ?? 0,
    rank: idx + 1,
    page: derivePageFromSource(s),
  }));
}

export function randomId() {
  return Math.random().toString(36).slice(2, 8);
}

export function formatRelativeTime(isoString) {
  const then = new Date(isoString).getTime();
  const diffSec = Math.max(0, Math.floor((Date.now() - then) / 1000));

  if (diffSec < 60) return "just now";
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 7) return `${diffDay}d ago`;
  return new Date(isoString).toLocaleDateString();
}

export function pairMessagesIntoEntries(messages = []) {
  const entries = [];
  let current = null;

  for (const message of messages) {
    if (message.role === "user") {
      if (current) entries.push(current);
      current = { query: message.content, answer: "", sources: [], confidence: null, pipeline: [], cacheHit: false };
    } else if (message.role === "assistant") {
      if (current) {
        current.answer = message.content;
        entries.push(current);
        current = null;
      } else {
        entries.push({ query: "", answer: message.content, sources: [], confidence: null, pipeline: [], cacheHit: false });
      }
    }
  }
  if (current) entries.push(current);

  return entries;
}