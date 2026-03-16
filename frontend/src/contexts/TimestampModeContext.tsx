import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import type { TimestampMode } from '../utils/timestamp';
import { TimestampModeContext } from './timestampModeContextValue';
import type { TimestampModeContextValue } from './timestampModeContextValue';

const STORAGE_KEY = 'podly.timestamp.mode';

function loadStoredMode(): TimestampMode {
  if (typeof window === 'undefined') {
    return 'local';
  }

  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === 'utc' || stored === 'local') {
      return stored;
    }
  } catch {
    // ignore storage failures
  }

  return 'local';
}

export function TimestampModeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<TimestampMode>(() => loadStoredMode());

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    try {
      window.localStorage.setItem(STORAGE_KEY, mode);
    } catch {
      // ignore storage failures
    }
  }, [mode]);

  const toggleMode = useCallback(() => {
    setMode((prev) => (prev === 'local' ? 'utc' : 'local'));
  }, []);

  const value = useMemo<TimestampModeContextValue>(
    () => ({ mode, setMode, toggleMode }),
    [mode, toggleMode]
  );

  return (
    <TimestampModeContext.Provider value={value}>
      {children}
    </TimestampModeContext.Provider>
  );
}
