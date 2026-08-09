import { Card } from "../ui";

const DIFFERENTIATORS = [
  { symbol: "⟳", title: "Adaptive Query Rewriting", desc: "Rewrites queries using single or multi-query strategies based on the planner's intent assessment." },
  { symbol: "⚡", title: "Semantic + Response Caching", desc: "Redis-backed caching at both the semantic and full-response levels to skip redundant LLM calls." },
  { symbol: "◈", title: "Vector Retrieval with Scores", desc: "Qdrant-powered similarity search returns top-k documents with cosine scores exposed in the UI." },
  { symbol: "↑",  title: "Cross-Encoder Reranking", desc: "A dedicated reranker model re-scores retrieved documents to surface the most relevant context." },
  { symbol: "◎", title: "Confidence-Based Routing", desc: "Evaluator node scores retrieval quality and routes to fallback LLM generation when confidence is low." },
  { symbol: "⬡", title: "LangGraph Execution Graph", desc: "Explicit node graph: pre_planner → rewrite → retrieve → evaluator → rerank → generate. Fully inspectable." },
  { symbol: "⋈", title: "Hybrid Dense + Sparse Retrieval", desc: "Dense embeddings and BM25 sparse search fused with Reciprocal Rank Fusion, so lexical and semantic matches both surface." },
  { symbol: "⇄", title: "Multi-Provider Cost Routing", desc: "Groq handles high-frequency orchestration calls, OpenAI is reserved for user-facing generation, with automatic fallback if a provider fails." },
  { symbol: "✓", title: "Automated Evaluation Suite", desc: "Routing-trace assertions and Ragas-scored faithfulness/relevancy checks run against a golden set, with results tracked run over run." },
];

function CapabilitiesSection() {
  return (
    <section id="capabilities" className="max-w-4xl mx-auto px-6 py-20">
      <p className="text-[11px] font-mono text-emerald-500 tracking-widest uppercase mb-3">Capabilities</p>
      <h2 className="text-3xl font-semibold tracking-tight text-zinc-100 mb-10">What makes it different</h2>

      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {DIFFERENTIATORS.map(({ symbol, title, desc }) => (
          <Card key={title} className="p-5 space-y-3">
            <span className="text-lg text-zinc-500">{symbol}</span>
            <div>
              <p className="text-sm font-semibold text-zinc-200 mb-1">{title}</p>
              <p className="text-xs text-zinc-600 leading-relaxed">{desc}</p>
            </div>
          </Card>
        ))}
      </div>
    </section>
  );
}

export default CapabilitiesSection;
