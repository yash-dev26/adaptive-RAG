import { Card } from "../ui";

const NAIVE_NODES = ["Input", "Retrieve", "Output"];
const ADAPTIVE_NODES = ["Input", "Plan", "Rewrite", "Retrieve", "Evaluate", "Rerank", "Output"];
const ADAPTIVE_HIGHLIGHTED = new Set(["Plan", "Rewrite", "Evaluate", "Rerank"]);

function ArchitectureSection() {
  return (
    <section id="how-it-works" className="max-w-4xl mx-auto px-6 py-20">
      <p className="text-[11px] font-mono text-emerald-500 tracking-widest uppercase mb-3">Architecture</p>
      <h2 className="text-3xl font-semibold tracking-tight text-zinc-100 mb-2">Naive RAG vs. Adaptive RAG</h2>
      <p className="text-zinc-500 mb-10 max-w-lg leading-relaxed">
        Standard pipelines retrieve blindly. This one plans, rewrites, evaluates, and reranks before generating.
      </p>

      <div className="grid md:grid-cols-2 gap-4">
        <Card className="p-6">
          <p className="text-[10px] font-mono text-zinc-600 uppercase tracking-widest mb-4">Naive</p>
          <div className="flex items-center flex-wrap gap-2">
            {NAIVE_NODES.map((node, i) => (
              <span key={node} className="flex items-center gap-2">
                <span className="bg-zinc-800 text-zinc-400 text-xs font-mono px-3 py-1.5 rounded">{node}</span>
                {i < NAIVE_NODES.length - 1 && <span className="text-zinc-700 text-xs">→</span>}
              </span>
            ))}
          </div>
          <p className="text-sm text-zinc-600 mt-5 leading-relaxed">
            Fixed retrieval, no adaptation. Query goes straight to the vector DB; top-k results passed to LLM.
          </p>
        </Card>

        <Card className="p-6 border-emerald-500/20">
          <p className="text-[10px] font-mono text-emerald-500 uppercase tracking-widest mb-4">Adaptive</p>
          <div className="flex items-center flex-wrap gap-2">
            {ADAPTIVE_NODES.map((node, i) => (
              <span key={node} className="flex items-center gap-2">
                <span
                  className={[
                    "text-xs font-mono px-3 py-1.5 rounded",
                    ADAPTIVE_HIGHLIGHTED.has(node)
                      ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/25"
                      : "bg-zinc-800 text-zinc-400",
                  ].join(" ")}
                >
                  {node}
                </span>
                {i < ADAPTIVE_NODES.length - 1 && <span className="text-zinc-700 text-xs">→</span>}
              </span>
            ))}
          </div>
          <p className="text-sm text-zinc-600 mt-5 leading-relaxed">
            LangGraph-orchestrated with planner, multi-strategy rewriting, hybrid dense+sparse retrieval, confidence scoring, and cross-encoder reranking &mdash; routed across cost-aware LLM providers at every step.
          </p>
        </Card>
      </div>
    </section>
  );
}
export default ArchitectureSection;
