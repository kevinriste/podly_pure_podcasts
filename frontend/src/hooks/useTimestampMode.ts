import { useContext } from 'react';
import { TimestampModeContext } from '../contexts/timestampModeContextValue';
import type { TimestampModeContextValue } from '../contexts/timestampModeContextValue';

export function useTimestampMode(): TimestampModeContextValue {
  const context = useContext(TimestampModeContext);
  if (!context) {
    throw new Error('useTimestampMode must be used within TimestampModeProvider');
  }
  return context;
}
