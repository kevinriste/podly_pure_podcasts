export type TimestampMode = 'local' | 'utc';

const DATE_ONLY_RE = /^\d{4}-\d{2}-\d{2}$/;
const SPACE_DATETIME_RE = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?$/;
const TZ_SUFFIX_RE = /(z|[+-]\d{2}(?::?\d{2})?)$/i;
const OFFSET_NO_COLON_RE = /([+-]\d{2})(\d{2})$/;

function normalizeApiTimestamp(raw: string): string {
  let value = raw.trim();

  if (SPACE_DATETIME_RE.test(value)) {
    value = value.replace(' ', 'T');
  } else if (DATE_ONLY_RE.test(value)) {
    value = `${value}T00:00:00`;
  }

  if (OFFSET_NO_COLON_RE.test(value)) {
    value = value.replace(OFFSET_NO_COLON_RE, '$1:$2');
  }

  if (!TZ_SUFFIX_RE.test(value)) {
    value = `${value}Z`;
  }

  return value;
}

export function parseApiTimestamp(
  input: string | number | Date
): Date {
  if (input instanceof Date) {
    return new Date(input.getTime());
  }

  if (typeof input === 'number') {
    return new Date(input);
  }

  return new Date(normalizeApiTimestamp(input));
}

function withModeTimezone(
  mode: TimestampMode,
  options: Intl.DateTimeFormatOptions
): Intl.DateTimeFormatOptions {
  if (mode === 'utc') {
    return {
      ...options,
      timeZone: 'UTC',
      hour12: false,
      hourCycle: 'h23',
    };
  }
  return options;
}

export function formatTimestampDateTime(
  input: string | number | Date | null | undefined,
  mode: TimestampMode,
  options?: Intl.DateTimeFormatOptions
): string {
  if (input === null || input === undefined || input === '') {
    return '—';
  }

  const date = parseApiTimestamp(input);
  if (Number.isNaN(date.getTime())) {
    return String(input);
  }

  const formatted = date.toLocaleString(
    undefined,
    withModeTimezone(mode, options ?? {})
  );

  return mode === 'utc' ? `${formatted} UTC` : formatted;
}

export function formatTimestampDate(
  input: string | number | Date | null | undefined,
  mode: TimestampMode,
  options?: Intl.DateTimeFormatOptions
): string {
  if (input === null || input === undefined || input === '') {
    return '—';
  }

  const date = parseApiTimestamp(input);
  if (Number.isNaN(date.getTime())) {
    return String(input);
  }

  return date.toLocaleDateString(
    undefined,
    withModeTimezone(mode, options ?? {})
  );
}
