import { Badge } from "../ui";

const TECH_STACK = [
  { name: "FastAPI",    color: "#14b8a6", desc: "Async REST backend" },
  { name: "LangGraph",  color: "#8b5cf6", desc: "Execution graph" },
  { name: "Qdrant",     color: "#ef4444", desc: "Vector database" },
  { name: "Redis",      color: "#f97316", desc: "Semantic cache" },
  { name: "MongoDB",    color: "#22c55e", desc: "Checkpointing" },
  { name: "Cohere",     color: "#6366f1", desc: "Reranking" },
  { name: "Tavily",     color: "#06b6d4", desc: "Web search fallback" },
];

function TechStackSection() {
  return (
    <section className="max-w-4xl mx-auto px-6 py-20">
      <p className="text-[11px] font-mono text-emerald-500 tracking-widest uppercase mb-6">Stack</p>
      <div className="flex flex-wrap gap-3">
        {TECH_STACK.map(({ name, color, desc }) => (
          <div
            key={name}
            className="flex items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-2.5"
          >
            <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
            <span className="font-mono text-sm text-zinc-200">{name}</span>
            <span className="text-zinc-700 text-xs">— {desc}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

export default TechStackSection;
