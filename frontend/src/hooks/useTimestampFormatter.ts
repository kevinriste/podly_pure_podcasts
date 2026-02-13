import { useCallback, useMemo } from 'react';
import { useTimestampMode } from './useTimestampMode';
import { formatTimestampDate, formatTimestampDateTime } from '../utils/timestamp';

export function useTimestampFormatter() {
  const { mode } = useTimestampMode();

  const formatDate = useCallback(
    (value: string | number | Date | null | undefined, options?: Intl.DateTimeFormatOptions) =>
      formatTimestampDate(value, mode, options),
    [mode]
  );

  const formatDateTime = useCallback(
    (value: string | number | Date | null | undefined, options?: Intl.DateTimeFormatOptions) =>
      formatTimestampDateTime(value, mode, options),
    [mode]
  );

  return useMemo(
    () => ({
      mode,
      formatDate,
      formatDateTime,
    }),
    [mode, formatDate, formatDateTime]
  );
}
