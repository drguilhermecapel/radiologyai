import { useState } from "react";
import type { FormEvent, ReactNode } from "react";

import { useI18n } from "../i18n/i18n";
import { useSession } from "../session";

export function LoginPage(): ReactNode {
  const { t } = useI18n();
  const { login } = useSession();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent): Promise<void> => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(username, password);
    } catch (err) {
      setError(t("login.failed", { detail: err instanceof Error ? err.message : String(err) }));
    } finally {
      setBusy(false);
      setPassword("");
    }
  };

  return (
    <section className="login card" aria-labelledby="login-title">
      <h1 id="login-title">{t("login.title")}</h1>
      <form onSubmit={(e) => void submit(e)}>
        <label className="field">
          <span>{t("login.username")}</span>
          <input
            name="username"
            autoComplete="username"
            required
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />
        </label>
        <label className="field">
          <span>{t("login.password")}</span>
          <input
            name="password"
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>
        {error ? (
          <p className="error-box" role="alert">
            {error}
          </p>
        ) : null}
        <button type="submit" className="btn btn-primary" disabled={busy}>
          {busy ? t("login.busy") : t("login.submit")}
        </button>
      </form>
      <p className="muted small">{t("login.noDefault")}</p>
    </section>
  );
}
