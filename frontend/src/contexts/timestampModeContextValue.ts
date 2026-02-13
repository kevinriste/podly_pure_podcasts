import { createContext } from 'react';
import type { TimestampMode } from '../utils/timestamp';

export interface TimestampModeContextValue {
  mode: TimestampMode;
  setMode: (mode: TimestampMode) => void;
  toggleMode: () => void;
}

export const TimestampModeContext = createContext<TimestampModeContextValue | undefined>(undefined);
