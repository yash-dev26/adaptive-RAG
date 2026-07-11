import { createContext, useCallback, useContext, useMemo, useState } from "react";
import { getApiKeys, setApiKeys as persistApiKeys, clearApiKeys } from "../lib/apiKeys.js";

const ApiKeysContext = createContext(null);

export function ApiKeysProvider({ children }) {
  const [keys, setKeys] = useState(() => getApiKeys());
  const [isModalOpen, setIsModalOpen] = useState(false);

  const saveKeys = useCallback((next) => {
    persistApiKeys(next);
    setKeys(getApiKeys());
  }, []);

  const clearKeys = useCallback(() => {
    clearApiKeys();
    setKeys(getApiKeys());
  }, []);

  const openModal = useCallback(() => setIsModalOpen(true), []);
  const closeModal = useCallback(() => setIsModalOpen(false), []);

  const value = useMemo(
    () => ({
      openaiKey: keys.openaiKey,
      groqKey: keys.groqKey,
      hasOpenAiKey: Boolean(keys.openaiKey),
      isModalOpen,
      openModal,
      closeModal,
      saveKeys,
      clearKeys,
    }),
    [keys, isModalOpen, openModal, closeModal, saveKeys, clearKeys]
  );

  return <ApiKeysContext.Provider value={value}>{children}</ApiKeysContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useApiKeys() {
  const ctx = useContext(ApiKeysContext);
  if (!ctx) {
    throw new Error("useApiKeys must be used within an ApiKeysProvider");
  }
  return ctx;
}