import { Link } from "react-router-dom";
import { Button, Badge } from "./ui";
import { useApiKeys } from "../context/ApiKeysContext.jsx";

const NAV_LINKS = [
  { label: "Architecture", href: "#how-it-works" },
  { label: "Capabilities", href: "#capabilities" },
  { label: "GitHub", href: "https://github.com/yash-dev26/adaptive-rag", external: true },
];

export default function Nav() {
  const { hasOpenAiKey, openModal } = useApiKeys();

  return (
    <nav className="fixed top-0 inset-x-0 z-50 h-14 border-b border-zinc-800 bg-zinc-950/90 backdrop-blur-sm">
      <div className="max-w-6xl mx-auto h-full px-6 flex items-center justify-between">
        <Link to="/" className="font-mono text-sm font-semibold text-zinc-100 tracking-tight no-underline">
          ADAPTIVE RAG
        </Link>

        <div className="flex items-center gap-6">
          {NAV_LINKS.map(({ label, href, external }) => (
            <a
              key={label}
              href={href}
              target={external ? "_blank" : undefined}
              rel={external ? "noreferrer" : undefined}
              className="text-sm text-zinc-500 hover:text-zinc-200 transition-colors no-underline"
            >
              {label}
            </a>
          ))}

          <Button variant="outline" size="sm" className="font-mono gap-2" onClick={openModal}>
            API Keys
            {hasOpenAiKey ? <Badge variant="accent">set</Badge> : <Badge variant="warning">needed</Badge>}
          </Button>

          <Link to="/chat">
            <Button variant="accent" size="sm" className="font-mono">
              Open chat →
            </Button>
          </Link>
        </div>
      </div>
    </nav>
  );
}
