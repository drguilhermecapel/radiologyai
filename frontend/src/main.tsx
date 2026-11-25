import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { I18nProvider } from "./i18n/i18n";
import { SessionProvider } from "./session";
import "./styles.css";

const root = document.getElementById("root");
if (root === null) throw new Error("elemento #root ausente em index.html");

createRoot(root).render(
  <StrictMode>
    <I18nProvider>
      <SessionProvider>
        <App />
      </SessionProvider>
    </I18nProvider>
  </StrictMode>,
);
