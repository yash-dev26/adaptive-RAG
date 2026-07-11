import { Link } from "react-router-dom";
import { Button } from "../ui";


function CtaSection() {
  return (
    <section className="max-w-4xl mx-auto px-6 py-24 text-center">
      <h2 className="text-3xl font-semibold tracking-tight text-zinc-100 mb-3">Ready to try it?</h2>
      <p className="text-zinc-500 mb-8 max-w-sm mx-auto">Bring your own OpenAI key, upload a PDF, and ask questions with full pipeline transparency.</p>
      <Link to="/chat">
        <Button variant="accent" size="lg" className="font-mono">
          Open chat →
        </Button>
      </Link>
    </section>
  );
}
export default CtaSection;
