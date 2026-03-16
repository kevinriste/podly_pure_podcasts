import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { feedsApi } from '../services/api';

interface ReprocessButtonProps {
  episodeGuid: string;
  isWhitelisted: boolean;
  feedId?: number;
  canModifyEpisodes?: boolean;
  className?: string;
  onReprocessStart?: () => void;
}

interface ReprocessInfo {
  existing_whisper_model: string | null;
  current_whisper_model: string | null;
  model_changed: boolean;
  has_transcript: boolean;
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
  const [error, setError] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [regenerateTranscript, setRegenerateTranscript] = useState(false);
  const [reprocessInfo, setReprocessInfo] = useState<ReprocessInfo | null>(null);
  const [loadingInfo, setLoadingInfo] = useState(false);
  const queryClient = useQueryClient();

  const handleReprocessClick = async () => {
    if (!isWhitelisted) {
      setError('Post must be whitelisted before reprocessing');
      return;
    }

    setRegenerateTranscript(false);
    setLoadingInfo(true);
    setShowModal(true);

    try {
      const info = await feedsApi.getReprocessInfo(episodeGuid);
      setReprocessInfo(info);
      // Auto-check regenerate if model changed
      if (info.model_changed && info.has_transcript) {
        setRegenerateTranscript(true);
      }
    } catch {
      setReprocessInfo(null);
    } finally {
      setLoadingInfo(false);
    }
  };

  const handleConfirmReprocess = async () => {
    setShowModal(false);
    setIsReprocessing(true);
    setError(null);

    try {
      const response = await feedsApi.reprocessPost(episodeGuid, {
        force_retranscribe: regenerateTranscript,
      });

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
        ? (err as { response?: { data?: { message?: string } } }).response?.data?.message || 'Failed to start reprocessing'
        : 'Failed to start reprocessing';
      setError(errorMessage);
    } finally {
      setIsReprocessing(false);
    }
  };

  if (!isWhitelisted || !canModifyEpisodes) {
    return null;
  }

  return (
    <div className={`${className}`}>
      <button
        onClick={handleReprocessClick}
        disabled={isReprocessing}
        className={`px-3 py-1 text-xs rounded font-medium transition-colors border ${
          isReprocessing
            ? 'bg-gray-500 text-white cursor-wait border-gray-500'
            : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50 hover:border-gray-400 hover:text-gray-900'
        }`}
        title={
          isReprocessing
            ? 'Clearing data and reprocessing...'
            : 'Clear all processing data and start fresh processing'
        }
      >
        {isReprocessing ? (
          '⏳ Reprocessing...'
        ) : (
          'Reprocess'
        )}
      </button>

      {error && (
        <div className="text-xs text-red-600 mt-1">
          {error}
        </div>
      )}

      {/* Confirmation Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-md w-full overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b">
              <h2 className="text-xl font-bold text-gray-900">Confirm Reprocess</h2>
              <button
                onClick={() => setShowModal(false)}
                className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Content */}
            <div className="p-6">
              <p className="text-gray-700 mb-4">
                Are you sure you want to reprocess this episode? This will delete the existing processed data and start fresh processing.
              </p>

              {/* Whisper model change notice */}
              {loadingInfo && (
                <div className="text-sm text-gray-500 mb-4">Loading transcript info...</div>
              )}
              {reprocessInfo?.model_changed && reprocessInfo.has_transcript && (
                <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-md text-sm">
                  <p className="font-medium text-amber-800 mb-1">Whisper model has changed</p>
                  <p className="text-amber-700">
                    This episode was transcribed with{' '}
                    <span className="font-mono text-xs bg-amber-100 px-1 rounded">
                      {reprocessInfo.existing_whisper_model}
                    </span>
                    {' '}but the current model is{' '}
                    <span className="font-mono text-xs bg-amber-100 px-1 rounded">
                      {reprocessInfo.current_whisper_model}
                    </span>.
                  </p>
                </div>
              )}

              <label className="flex items-start gap-2 mb-6 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={regenerateTranscript}
                  onChange={(event) => setRegenerateTranscript(event.target.checked)}
                />
                <span>
                  Also regenerate transcript
                  <span className="block text-xs text-gray-500">
                    {reprocessInfo?.model_changed && reprocessInfo.has_transcript
                      ? 'Pre-checked because the whisper model has changed. Uncheck to keep the existing transcript.'
                      : 'Disabled by default. Enable this to force a full re-transcription.'}
                  </span>
                </span>
              </label>

              {/* Action Buttons */}
              <div className="flex gap-3 justify-end">
                <button
                  onClick={() => setShowModal(false)}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 hover:border-gray-400 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleConfirmReprocess}
                  className="px-4 py-2 text-sm font-medium text-white bg-orange-600 rounded-md hover:bg-orange-700 transition-colors"
                >
                  Reprocess Episode
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
