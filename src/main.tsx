import "@fontsource/outfit/400.css";
import "@fontsource/outfit/500.css";
import "@fontsource/outfit/700.css";
import { createRoot } from "react-dom/client";
import App from "./App";
import ErrorBoundary from "./hud/ErrorBoundary";
import "./index.css";

// No StrictMode: its dev double-mount would churn WebGL contexts, and this
// is a kiosk app with one long-lived scene.
createRoot(document.getElementById("root")!).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);
