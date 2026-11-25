import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { LANGS, format, type Lang, type MessageKey } from "./messages";

const LANG_KEY = "radiologyai.lang";

export type TFunction = (key: MessageKey, vars?: Record<string, string | number>) => string;

interface I18nValue {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: TFunction;
  /** Data/hora no idioma corrente. */
  dt: (iso: string | null | undefined) => string;
}

const I18nContext = createContext<I18nValue | null>(null);

function initialLang(): Lang {
  try {
    const stored = localStorage.getItem(LANG_KEY);
    if (stored === "pt" || stored === "en") return stored;
  } catch {
    // sem armazenamento: padrão PT-BR
  }
  return "pt";
}

export function I18nProvider({ children }: { children: ReactNode }): ReactNode {
  const [lang, setLangState] = useState<Lang>(initialLang);

  useEffect(() => {
    document.documentElement.lang = lang === "pt" ? "pt-BR" : "en";
  }, [lang]);

  const setLang = useCallback((next: Lang) => {
    setLangState(next);
    try {
      localStorage.setItem(LANG_KEY, next);
    } catch {
      // preferência só nesta sessão
    }
  }, []);

  const value = useMemo<I18nValue>(() => {
    const messages = LANGS[lang];
    const locale = lang === "pt" ? "pt-BR" : "en-GB";
    const fmt = new Intl.DateTimeFormat(locale, { dateStyle: "short", timeStyle: "short" });
    return {
      lang,
      setLang,
      t: (key, vars) => format(messages[key], vars),
      dt: (iso) => {
        if (!iso) return "—";
        const d = new Date(iso);
        return Number.isNaN(d.getTime()) ? iso : fmt.format(d);
      },
    };
  }, [lang, setLang]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nValue {
  const ctx = useContext(I18nContext);
  if (ctx === null) throw new Error("useI18n fora de I18nProvider");
  return ctx;
}
