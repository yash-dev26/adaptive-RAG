import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";
import { ApiKeysProvider } from "./context/ApiKeysContext.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <ApiKeysProvider>
      <App />
    </ApiKeysProvider>
  </StrictMode>
);
