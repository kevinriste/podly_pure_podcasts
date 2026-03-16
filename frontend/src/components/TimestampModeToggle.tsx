import { useTimestampMode } from '../hooks/useTimestampMode';

interface TimestampModeToggleProps {
  className?: string;
}

export default function TimestampModeToggle({ className = '' }: TimestampModeToggleProps) {
  const { mode, toggleMode } = useTimestampMode();
  const isUtc = mode === 'utc';

  return (
    <button
      type="button"
      onClick={toggleMode}
      className={`inline-flex items-center rounded-md border border-gray-300 px-2 py-1 text-xs font-medium text-gray-700 hover:bg-gray-100 ${className}`}
      title={isUtc ? 'Switch timestamps to local time' : 'Switch timestamps to UTC'}
    >
      Time: {isUtc ? 'UTC' : 'Local'}
    </button>
  );
}
