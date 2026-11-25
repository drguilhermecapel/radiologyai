/**
 * Sessão do usuário: token, ``/auth/me`` (permissões vindas do servidor) e o
 * interruptor do modo pesquisa (só para papéis com ``research:view``).
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { api, getToken, onTokenChange, setToken, unwrap, type UserOut } from "./api/client";

export type Permission =
  | "users:manage"
  | "studies:read"
  | "studies:upload"
  | "studies:set_exam_type"
  | "analyses:create"
  | "analyses:read"
  | "research:view"
  | "reviews:write"
  | "reports:create"
  | "reports:read"
  | "reports:sign"
  | "models:read"
  | "catalog:read"
  | "audit:read"
  | "dicomweb:import"
  | "dicomweb:export"
  | "legacy:analyze";

interface SessionValue {
  user: UserOut | null;
  loading: boolean;
  can: (p: Permission) => boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  research: boolean;
  setResearch: (on: boolean) => void;
}

const SessionContext = createContext<SessionValue | null>(null);

export function SessionProvider({ children }: { children: ReactNode }): ReactNode {
  const [user, setUser] = useState<UserOut | null>(null);
  const [loading, setLoading] = useState<boolean>(getToken() !== null);
  const [research, setResearchState] = useState(false);

  const refresh = useCallback(async () => {
    if (getToken() === null) {
      setUser(null);
      setResearchState(false);
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      setUser(unwrap(await api.GET("/api/v1/auth/me")));
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    return onTokenChange(() => void refresh());
  }, [refresh]);

  const can = useCallback(
    (p: Permission) => user !== null && user.permissions.includes(p),
    [user],
  );

  const login = useCallback(async (username: string, password: string) => {
    const out = unwrap(await api.POST("/api/v1/auth/login", { body: { username, password } }));
    setUser(out.user);
    setToken(out.token);
  }, []);

  const logout = useCallback(async () => {
    try {
      await api.POST("/api/v1/auth/logout");
    } finally {
      setToken(null);
    }
  }, []);

  const setResearch = useCallback(
    (on: boolean) => setResearchState(on && user !== null && user.permissions.includes("research:view")),
    [user],
  );

  const value = useMemo<SessionValue>(
    () => ({ user, loading, can, login, logout, research: research && can("research:view"), setResearch }),
    [user, loading, can, login, logout, research, setResearch],
  );
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionValue {
  const ctx = useContext(SessionContext);
  if (ctx === null) throw new Error("useSession fora de SessionProvider");
  return ctx;
}
