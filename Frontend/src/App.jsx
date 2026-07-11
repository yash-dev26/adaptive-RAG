import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Nav from "./components/Nav";
import LandingPage from "./pages/LandingPage";
import ChatPage from "./pages/ChatPage";
import ApiKeyModal from "./components/ApiKeyModal";
import RequireApiKey from "./components/requireApiKey";
import { useEffect } from "react";

export default function App() {

    useEffect(() => {
        const wakeBackend = async () => {
            try {
            await fetch(`${import.meta.env.VITE_API_BASE_URL}/health/`, {
                method: "GET",
            });

            console.log("Backend wakeup ping sent.");
            } catch (err) {
            console.log("Backend wakeup failed:", err);
            }
        };

        wakeBackend();
    }, []);

  return (
    <Router>
      <div className="bg-zinc-950 text-zinc-100 min-h-screen flex flex-col">
        <Routes>
          <Route
            path="/"
            element={
              <>
                <Nav />
                <LandingPage />
              </>
            }
          />
          <Route
            path="/chat"
            element={
              <RequireApiKey>
                <ChatPage />
              </RequireApiKey>
            }
          />
        </Routes>
        <ApiKeyModal />
      </div>
    </Router>
  );
}