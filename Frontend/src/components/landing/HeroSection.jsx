import { Link } from "react-router-dom";
import { Badge, Button } from "../ui";

function HeroSection() {
  return (
    <section className="max-w-4xl mx-auto px-6 pt-32 pb-24">
      <div className="mb-4">
        <Badge variant="accent">RAG with full pipeline visibility</Badge>
      </div>
      <h1 className="text-5xl md:text-6xl font-semibold tracking-tight text-zinc-100 mb-6 leading-none">
        Adaptive RAG<br />
        <span className="text-zinc-500">that thinks before it retrieves.</span>
      </h1>
      <p className="text-lg text-zinc-500 max-w-xl leading-relaxed mb-10">
        A LangGraph-orchestrated pipeline that plans, rewrites, evaluates, and reranks
        before generating — with every decision exposed in the UI.
      </p>
      <div className="flex items-center gap-3">
        <Link to="/chat">
          <Button variant="accent" size="lg" className="font-mono">
            Get started →
          </Button>
        </Link>
        <a href="https://github.com/yash-dev26/adaptive-rag" target="_blank" rel="noreferrer">
          <Button variant="outline" size="lg" className="font-mono">
            GitHub
          </Button>
        </a>
      </div>
    </section>
  );
}
export default HeroSection;
