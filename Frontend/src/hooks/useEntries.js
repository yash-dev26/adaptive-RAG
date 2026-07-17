import { useState, useCallback } from "react";

export function useEntries() {
  const [entries, setEntries] = useState([]);

  const addEntry = useCallback((entry) => {
    setEntries((prev) => [...prev, entry]);
  }, []);

  const resetEntries = useCallback(() => {
    setEntries([]);
  }, []);

  return { entries, setEntries, addEntry, resetEntries };
}