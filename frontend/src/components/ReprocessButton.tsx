import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { feedsApi } from '../services/api';

type ReprocessMode = 'full' | 'keep-transcript';

interface ReprocessButtonProps {
  episodeGuid: string;
  isWhitelisted: boolean;
  feedId?: number;
  canModifyEpisodes?: boolean;
  className?: string;
  onReprocessStart?: () => void;
}

export default function ReprocessButton({
  episodeGuid,
  isWhitelisted,
  feedId,
  canModifyEpisodes = true,
  className = '',
  onReprocessStart
}: ReprocessButtonProps) {
  const [isReprocessing, setIsReprocessing] = useState(false);
  const [activeMode, setActiveMode] = useState<ReprocessMode | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showModalMode, setShowModalMode] = useState<ReprocessMode | null>(null);
  const queryClient = useQueryClient();

  const handleReprocessClick = async (mode: ReprocessMode) => {
    if (!isWhitelisted) {
      setError('Post must be whitelisted before reprocessing');
      return;
    }

    setShowModalMode(mode);
  };

  const handleConfirmReprocess = async () => {
    const mode = showModalMode;
    if (!mode) {
      return;
    }

    setShowModalMode(null);
    setIsReprocessing(true);
    setActiveMode(mode);
    setError(null);

    try {
      const response = mode === 'keep-transcript'
        ? await feedsApi.reprocessPostKeepTranscript(episodeGuid)
        : await feedsApi.reprocessPost(episodeGuid);

      if (response.status === 'started') {
        // Notify parent component that reprocessing started
        onReprocessStart?.();

        // Invalidate queries to refresh the UI
        if (feedId) {
          queryClient.invalidateQueries({ queryKey: ['episodes', feedId] });
        }
        queryClient.invalidateQueries({ queryKey: ['episode-stats', episodeGuid] });
      } else {
        setError(response.message || 'Failed to start reprocessing');
      }
    } catch (err: unknown) {
      console.error('Error starting reprocessing:', err);
      const errorMessage = err && typeof err === 'object' && 'response' in err
        ? (err as { response?: { data?: { message?: string } } }).response?.data?.message
          || 'Failed to start reprocessing'
        : 'Failed to start reprocessing';
      setError(errorMessage);
    } finally {
      setIsReprocessing(false);
      setActiveMode(null);
    }
  };

  if (!isWhitelisted || !canModifyEpisodes) {
    return null;
  }

  const isKeepTranscriptMode = showModalMode === 'keep-transcript';
  const modalTitle = isKeepTranscriptMode
    ? 'Confirm Reprocess (Keep Transcript)'
    : 'Confirm Reprocess';
  const modalMessage = isKeepTranscriptMode
    ? 'Are you sure you want to reprocess this episode while keeping the existing transcript? This will delete processed outputs and restart processing after transcription (if the existing transcript is reusable).'
    : 'Are you sure you want to reprocess this episode? This will delete the existing processed data and start fresh processing.';
  const confirmButtonLabel = isKeepTranscriptMode
    ? 'Reprocess (Keep Transcript)'
    : 'Reprocess Episode';
  const keepTranscriptButtonTitle = isReprocessing && activeMode === 'keep-transcript'
    ? 'Reprocessing from transcript stage...'
    : 'Reprocess while preserving the existing transcript and restarting after transcription';

  return (
    <div className={`${className}`}>
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => handleReprocessClick('full')}
          disabled={isReprocessing}
          className={`px-3 py-1 text-xs rounded font-medium transition-colors border ${
            isReprocessing
              ? 'bg-gray-500 text-white cursor-wait border-gray-500'
              : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50 hover:border-gray-400 hover:text-gray-900'
          }`}
          title={
            isReprocessing && activeMode === 'full'
              ? 'Clearing data and reprocessing...'
              : 'Clear all processing data and start fresh processing'
          }
        >
          {isReprocessing && activeMode === 'full' ? '⏳ Reprocessing...' : 'Reprocess'}
        </button>

        <button
          onClick={() => handleReprocessClick('keep-transcript')}
          disabled={isReprocessing}
          className={`px-3 py-1 text-xs rounded font-medium transition-colors border ${
            isReprocessing
              ? 'bg-gray-500 text-white cursor-wait border-gray-500'
              : 'bg-white text-blue-700 border-blue-300 hover:bg-blue-50 hover:border-blue-400'
          }`}
          title={keepTranscriptButtonTitle}
        >
          {isReprocessing && activeMode === 'keep-transcript' ? (
            <>
              <span className="sm:hidden">⏳ Reusing...</span>
              <span className="hidden sm:inline">⏳ Reprocessing (Transcript)...</span>
            </>
          ) : (
            <>
              <span className="sm:hidden">Reuse Transcript</span>
              <span className="hidden sm:inline">Reprocess (Keep Transcript)</span>
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="text-xs text-red-600 mt-1">
          {error}
        </div>
      )}

      {/* Confirmation Modal */}
      {showModalMode && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-md w-full overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b">
              <h2 className="text-xl font-bold text-gray-900">{modalTitle}</h2>
              <button
                onClick={() => setShowModalMode(null)}
                className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Content */}
            <div className="p-6">
              <p className="text-gray-700 mb-6">
                {modalMessage}
              </p>

              {/* Action Buttons */}
              <div className="flex gap-3 justify-end">
                <button
                  onClick={() => setShowModalMode(null)}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 hover:border-gray-400 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleConfirmReprocess}
                  className={`px-4 py-2 text-sm font-medium text-white rounded-md transition-colors ${
                    isKeepTranscriptMode
                      ? 'bg-blue-600 hover:bg-blue-700'
                      : 'bg-orange-600 hover:bg-orange-700'
                  }`}
                >
                  {confirmButtonLabel}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
