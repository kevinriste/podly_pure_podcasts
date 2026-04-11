import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { feedsApi } from '../services/api';
import { useTimestampFormatter } from '../hooks/useTimestampFormatter';

interface LLMProcessingStatsProps {
  episodeGuid: string;
  isStatsReady: boolean;
  className?: string;
}

type TabId = 'overview' | 'model-calls' | 'transcript' | 'identifications' | 'audio' | 'speakers';

type TranscriptRow =
  | { type: 'transcript'; data: { id: number; sequence_num: number; start_time: number; end_time: number; text: string; speaker?: string | null; primary_label: 'ad' | 'content'; mixed: boolean; identifications: Array<{ id: number; label: string; confidence: number | null; model_call_id: number }> } }
  | { type: 'audio_marker'; data: { id: number; start_time: number; end_time: number; label: string } };

type IdentificationRow =
  | { type: 'identification'; data: { id: number; transcript_segment_id: number; label: string; confidence: number | null; model_call_id: number; segment_sequence_num: number; segment_start_time: number; segment_end_time: number; segment_text: string; segment_speaker?: string | null; mixed: boolean } }
  | { type: 'audio_marker'; data: { id: number; start_time: number; end_time: number; label: string } };

interface SpeakerStat {
  speaker: string;
  segmentCount: number;
  totalTime: number;
  wordCount: number;
  percentOfTotal: number;
}

export default function LLMProcessingStats({
  episodeGuid,
  isStatsReady,
  className = ''
}: LLMProcessingStatsProps) {
  const { formatDateTime } = useTimestampFormatter();
  const [showModal, setShowModal] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [expandedModelCalls, setExpandedModelCalls] = useState<Set<number>>(new Set());

  const { data: stats, isLoading, error } = useQuery({
    queryKey: ['episode-stats', episodeGuid],
    queryFn: () => feedsApi.getPostStats(episodeGuid),
    enabled: showModal && isStatsReady,
  });

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.round(seconds % 60);

    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    }
    return `${minutes}m ${secs}s`;
  };

  const formatTimelineLabel = (seconds: number) => {
    const totalSeconds = Math.max(0, Math.round(seconds));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const secs = totalSeconds % 60;

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  const formatBytes = (bytes: number | null) => {
    if (bytes === null || Number.isNaN(bytes)) return 'unknown';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const toggleModelCallDetails = (callId: number) => {
    const newExpanded = new Set(expandedModelCalls);
    if (newExpanded.has(callId)) {
      newExpanded.delete(callId);
    } else {
      newExpanded.add(callId);
    }
    setExpandedModelCalls(newExpanded);
  };

  const formatConfidence = (value: number | null | undefined) => {
    if (value === null || value === undefined) return 'N/A';
    return value.toFixed(2);
  };

  const getTranscriptSegmentConfidence = (
    segment: {
      primary_label: 'ad' | 'content';
      identifications: Array<{ label: string; confidence: number | null }>;
    }
  ): number | null => {
    const adConfidences = segment.identifications
      .filter(i => i.label === 'ad' && i.confidence !== null && i.confidence !== undefined)
      .map(i => i.confidence as number);
    if (segment.primary_label === 'ad' && adConfidences.length > 0) return Math.max(...adConfidences);
    const allConfidences = segment.identifications
      .filter(i => i.confidence !== null && i.confidence !== undefined)
      .map(i => i.confidence as number);
    if (allConfidences.length === 0) return null;
    return Math.max(...allConfidences);
  };

  const getSpeakerColor = (speaker: string | null | undefined): string => {
    if (!speaker) return '';
    let hash = 0;
    for (let i = 0; i < speaker.length; i++) {
      hash = speaker.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = ((hash % 360) + 360) % 360;
    return `hsl(${hue}, 55%, 92%)`;
  };

  const getAudioLabelStyle = (label: string): { bg: string; text: string } => {
    switch (label) {
      case 'music': return { bg: 'bg-red-100', text: 'text-red-800' };
      case 'speech': return { bg: 'bg-blue-100', text: 'text-blue-800' };
      case 'noEnergy': return { bg: 'bg-gray-100', text: 'text-gray-800' };
      case 'noise': return { bg: 'bg-yellow-100', text: 'text-yellow-800' };
      default: return { bg: 'bg-gray-100', text: 'text-gray-600' };
    }
  };

  const hasAudioSegments = (stats?.audio_segments?.length ?? 0) > 0;

  const nonSpeechAudioSegments = useMemo(() => {
    if (!stats?.audio_segments) return [];
    return stats.audio_segments.filter(s => s.label !== 'speech');
  }, [stats?.audio_segments]);

  const mergedTranscriptRows: TranscriptRow[] = useMemo(() => {
    if (!stats) return [];
    const rows: TranscriptRow[] = (stats.transcript_segments || []).map(seg => ({
      type: 'transcript' as const,
      data: seg,
    }));
    for (const aseg of nonSpeechAudioSegments) {
      rows.push({ type: 'audio_marker' as const, data: aseg });
    }
    rows.sort((a, b) => {
      const aTime = a.type === 'transcript' ? a.data.start_time : a.data.start_time;
      const bTime = b.type === 'transcript' ? b.data.start_time : b.data.start_time;
      if (aTime !== bTime) return aTime - bTime;
      // Audio markers sort before transcript rows at the same time
      if (a.type === 'audio_marker' && b.type === 'transcript') return -1;
      if (a.type === 'transcript' && b.type === 'audio_marker') return 1;
      return 0;
    });
    return rows;
  }, [stats, nonSpeechAudioSegments]);

  const mergedIdentificationRows: IdentificationRow[] = useMemo(() => {
    if (!stats) return [];
    const rows: IdentificationRow[] = (stats.identifications || []).map(ident => ({
      type: 'identification' as const,
      data: ident,
    }));
    for (const aseg of nonSpeechAudioSegments) {
      rows.push({ type: 'audio_marker' as const, data: aseg });
    }
    rows.sort((a, b) => {
      const aTime = a.type === 'identification' ? a.data.segment_start_time : a.data.start_time;
      const bTime = b.type === 'identification' ? b.data.segment_start_time : b.data.start_time;
      if (aTime !== bTime) return aTime - bTime;
      if (a.type === 'audio_marker' && b.type === 'identification') return -1;
      if (a.type === 'identification' && b.type === 'audio_marker') return 1;
      return 0;
    });
    return rows;
  }, [stats, nonSpeechAudioSegments]);

  const speakerStats: SpeakerStat[] = useMemo(() => {
    if (!stats?.transcript_segments) return [];
    const map = new Map<string, { segmentCount: number; totalTime: number; wordCount: number }>();
    for (const seg of stats.transcript_segments) {
      const speaker = seg.speaker || null;
      if (!speaker) continue;
      const existing = map.get(speaker) || { segmentCount: 0, totalTime: 0, wordCount: 0 };
      existing.segmentCount += 1;
      existing.totalTime += Math.max(0, seg.end_time - seg.start_time);
      existing.wordCount += seg.text.trim().split(/\s+/).filter(w => w.length > 0).length;
      map.set(speaker, existing);
    }
    if (map.size === 0) return [];
    const totalTime = Array.from(map.values()).reduce((sum, s) => sum + s.totalTime, 0);
    const result: SpeakerStat[] = [];
    for (const [speaker, data] of map.entries()) {
      result.push({
        speaker,
        segmentCount: data.segmentCount,
        totalTime: data.totalTime,
        wordCount: data.wordCount,
        percentOfTotal: totalTime > 0 ? (data.totalTime / totalTime) * 100 : 0,
      });
    }
    result.sort((a, b) => b.totalTime - a.totalTime);
    return result;
  }, [stats?.transcript_segments]);

  const hasSpeakers = speakerStats.length > 0;

  const getAudioMarkerBg = (label: string): string => {
    switch (label) {
      case 'music': return 'bg-red-50';
      case 'noEnergy': return 'bg-gray-100';
      case 'noise': return 'bg-yellow-50';
      default: return 'bg-gray-50';
    }
  };

  const getAudioMarkerLabel = (label: string): string => {
    switch (label) {
      case 'music': return 'MUSIC';
      case 'noEnergy': return 'SILENCE';
      case 'noise': return 'NOISE';
      default: return label.toUpperCase();
    }
  };

  const getAudioMarkerTextColor = (label: string): string => {
    switch (label) {
      case 'music': return 'text-red-600';
      case 'noEnergy': return 'text-gray-500';
      case 'noise': return 'text-yellow-700';
      default: return 'text-gray-500';
    }
  };

  if (!isStatsReady) {
    return null;
  }

  return (
    <>
      <button
        onClick={() => setShowModal(true)}
        className={`px-3 py-1 text-xs rounded font-medium transition-colors border bg-white text-gray-700 border-gray-300 hover:bg-gray-50 hover:border-gray-400 hover:text-gray-900 flex items-center gap-1 ${className}`}
      >
        Stats
      </button>

      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-6xl w-full max-h-[90vh] overflow-hidden">
            <div className="flex items-center justify-between p-6 border-b">
              <h2 className="text-xl font-bold text-gray-900 text-left">Processing Statistics & Debug</h2>
              <button
                onClick={() => setShowModal(false)}
                className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="border-b">
              <nav className="flex space-x-8 px-6 overflow-x-auto scrollbar-thin">
                {[
                  { id: 'overview', label: 'Overview' },
                  { id: 'model-calls', label: 'Model Calls' },
                  { id: 'transcript', label: 'Transcript Segments' },
                  { id: 'identifications', label: 'Identifications' },
                  ...(hasAudioSegments ? [{ id: 'audio', label: 'Audio Segments' }] : []),
                  ...(hasSpeakers ? [{ id: 'speakers', label: 'Speakers' }] : []),
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as TabId)}
                    className={`py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap ${
                      activeTab === tab.id
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    {tab.label}
                    {stats && tab.id === 'model-calls' && stats.model_calls && ` (${stats.model_calls.length})`}
                    {stats && tab.id === 'transcript' && stats.transcript_segments && ` (${stats.transcript_segments.length})`}
                    {stats && tab.id === 'identifications' && stats.identifications && ` (${stats.identifications.length})`}
                    {stats && tab.id === 'audio' && stats.audio_segments && ` (${stats.audio_segments.length})`}
                    {stats && tab.id === 'speakers' && ` (${speakerStats.length})`}
                  </button>
                ))}
              </nav>
            </div>

            <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
              {isLoading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                  <span className="ml-3 text-gray-600">Loading stats...</span>
                </div>
              ) : error ? (
                <div className="text-center py-12">
                  <p className="text-red-600">Failed to load processing statistics</p>
                </div>
              ) : stats ? (
                <>
                  {activeTab === 'overview' && (
                    <div className="space-y-6">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h3 className="font-semibold text-gray-900 mb-2 text-left">Episode Information</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                          <div className="text-left">
                            <span className="font-medium text-gray-700">Title:</span>
                            <span className="ml-2 text-gray-600">{stats.post?.title || 'Unknown'}</span>
                          </div>
                          <div className="text-left">
                            <span className="font-medium text-gray-700">Duration:</span>
                            <span className="ml-2 text-gray-600">
                              {stats.post?.duration ? formatDuration(stats.post.duration) : 'Unknown'}
                            </span>
                          </div>
                          <div className="text-left">
                            <span className="font-medium text-gray-700">Detection Method:</span>
                            <span className="ml-2 px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800">
                              LLM Transcription
                            </span>
                          </div>
                        </div>
                      </div>

                      <div>
                        <h3 className="font-semibold text-gray-900 mb-4 text-left">Key Metrics</h3>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                          <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4 text-center">
                            <div className="text-2xl font-bold text-blue-600">
                              {stats.processing_stats?.total_segments || 0}
                            </div>
                            <div className="text-sm text-blue-800">Transcript Segments</div>
                          </div>

                          <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-4 text-center">
                            <div className="text-2xl font-bold text-green-600">
                              {stats.processing_stats?.content_segments || 0}
                            </div>
                            <div className="text-sm text-green-800">Content Segments</div>
                          </div>

                          <div className="bg-gradient-to-br from-red-50 to-red-100 rounded-lg p-4 text-center">
                            <div className="text-2xl font-bold text-red-600">
                              {stats.processing_stats?.ad_segments_count || 0}
                            </div>
                            <div className="text-sm text-red-800">Ad Segments Removed</div>
                          </div>
                        </div>
                      </div>

                      {stats.debug_info && (
                        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                          <h3 className="font-semibold text-gray-900 mb-2 text-left">Debug Details</h3>
                          <p className="text-xs text-amber-700 mb-4 text-left">
                            Visible because <code>PODLY_STATS_DEBUG</code> is enabled.
                          </p>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                            <div className="text-left">
                              <span className="font-medium text-gray-700">GUID:</span>
                              <span className="ml-2 text-gray-600 font-mono break-all">{stats.debug_info.guid}</span>
                            </div>
                            <div className="text-left">
                              <span className="font-medium text-gray-700">Post ID / Feed ID:</span>
                              <span className="ml-2 text-gray-600">{stats.debug_info.post_id} / {stats.debug_info.feed_id}</span>
                            </div>
                            <div className="text-left md:col-span-2">
                              <span className="font-medium text-gray-700">Download URL:</span>
                              <span className="ml-2 text-gray-600 font-mono break-all">{stats.debug_info.download_url}</span>
                            </div>
                            <div className="text-left md:col-span-2">
                              <span className="font-medium text-gray-700">Processed Audio Path:</span>
                              <span className="ml-2 text-gray-600 font-mono break-all">
                                {stats.debug_info.processed_audio.path || 'missing'}
                              </span>
                              <div className="text-xs text-gray-500 mt-1">
                                {stats.debug_info.processed_audio.exists
                                  ? `exists (${formatBytes(stats.debug_info.processed_audio.size_bytes)})`
                                  : 'missing'}
                              </div>
                            </div>
                            <div className="text-left md:col-span-2">
                              <span className="font-medium text-gray-700">Unprocessed Audio Path:</span>
                              <span className="ml-2 text-gray-600 font-mono break-all">
                                {stats.debug_info.unprocessed_audio.path || 'missing'}
                              </span>
                              <div className="text-xs text-gray-500 mt-1">
                                {stats.debug_info.unprocessed_audio.exists
                                  ? `exists (${formatBytes(stats.debug_info.unprocessed_audio.size_bytes)})`
                                  : 'missing'}
                              </div>
                            </div>
                            <div className="text-left md:col-span-2">
                              <span className="font-medium text-gray-700">Data Roots:</span>
                              <span className="ml-2 text-gray-600 font-mono break-all">
                                in: {stats.debug_info.processing_roots.in_root} | srv: {stats.debug_info.processing_roots.srv_root}
                              </span>
                            </div>
                            <div className="text-left">
                              <span className="font-medium text-gray-700">Record Counts:</span>
                              <span className="ml-2 text-gray-600">
                                segments {stats.debug_info.record_counts.transcript_segments}, calls {stats.debug_info.record_counts.model_calls}, ids {stats.debug_info.record_counts.identifications}
                              </span>
                            </div>
                          </div>

                          <div className="mt-4">
                            <h4 className="font-medium text-gray-900 mb-2 text-left">Processed Audio Path Candidates</h4>
                            {(stats.debug_info.processed_audio_path_candidates || []).length === 0 ? (
                              <p className="text-xs text-gray-500 text-left">No candidates derived.</p>
                            ) : (
                              <div className="space-y-2">
                                {(stats.debug_info.processed_audio_path_candidates || []).map((candidate, idx) => (
                                  <div key={`${candidate.path}-${idx}`} className="bg-white border border-amber-100 rounded p-2">
                                    <div className="font-mono text-xs text-gray-700 break-all text-left">{candidate.path}</div>
                                    <div className="text-xs text-gray-500 mt-1 text-left">
                                      {candidate.exists ? `exists (${formatBytes(candidate.size_bytes)})` : 'missing'}
                                      {candidate.error ? ` - ${candidate.error}` : ''}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {(() => {
                        const durationSeconds = (
                          stats.processing_stats?.original_duration_seconds
                          ?? ((stats.post?.duration ?? 0) + (stats.processing_stats?.estimated_ad_time_seconds ?? 0))
                        ) || (stats.transcript_segments?.length
                          ? Math.max(...stats.transcript_segments.map((segment) => segment.end_time))
                          : 0);
                        const fallbackAdBlocks = (() => {
                          const adSegments = (stats.transcript_segments || [])
                            .filter((segment) => segment.primary_label === 'ad')
                            .map((segment) => ({ start: segment.start_time, end: segment.end_time }))
                            .sort((a, b) => a.start - b.start);

                          if (!adSegments.length) return [];

                          const merged: Array<{ start: number; end: number }> = [];
                          let current = { ...adSegments[0] };
                          const gapSeconds = 1;
                          for (const segment of adSegments.slice(1)) {
                            if (segment.start <= current.end + gapSeconds) {
                              current.end = Math.max(current.end, segment.end);
                            } else {
                              merged.push(current);
                              current = { ...segment };
                            }
                          }
                          merged.push(current);
                          return merged;
                        })();

                        const apiAdBlocks = (stats.processing_stats?.ad_blocks || []).map((block) => ({
                          start: block.start_time,
                          end: block.end_time,
                        }));
                        const adBlocks = apiAdBlocks.length ? apiAdBlocks : fallbackAdBlocks;
                        const adTimeSeconds = stats.processing_stats?.estimated_ad_time_seconds
                          ?? adBlocks.reduce((sum, block) => sum + Math.max(0, block.end - block.start), 0);
                        const adPercent = durationSeconds > 0
                          ? (adTimeSeconds / durationSeconds) * 100
                          : 0;
                        const cleanSeconds = Math.max(0, durationSeconds - adTimeSeconds);
                        const timelineTicks = [0, 0.25, 0.5, 0.75, 1];

                        return (
                          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                            <h3 className="font-semibold text-gray-900 mb-4 text-left">Advertisement Removal Summary</h3>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
                              <div>
                                <div className="text-2xl font-bold text-blue-600">{adBlocks.length}</div>
                                <div className="text-sm text-gray-600">Ad Blocks</div>
                              </div>
                              <div>
                                <div className="text-2xl font-bold text-blue-600">{formatDuration(adTimeSeconds)}</div>
                                <div className="text-sm text-gray-600">Time Removed</div>
                              </div>
                              <div>
                                <div className="text-2xl font-bold text-rose-600">{adPercent.toFixed(1)}%</div>
                                <div className="text-sm text-gray-600">Episode Reduced</div>
                              </div>
                            </div>

                            <div className="mt-5 space-y-3">
                              <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-gray-600">
                                <div className="flex items-center gap-2">
                                  <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-white text-gray-500 border border-gray-200">
                                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                  </span>
                                  Episode Timeline
                                </div>
                                <div className="text-gray-600">
                                  {formatDuration(cleanSeconds)} clean
                                  <span className="text-rose-600 ml-2">
                                    {formatDuration(adTimeSeconds)} removed ({adPercent.toFixed(1)}%)
                                  </span>
                                </div>
                              </div>

                              <div className="relative h-3 w-full rounded-full bg-gray-200 overflow-hidden">
                                <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 via-blue-400/15 to-blue-500/20" />
                                {durationSeconds > 0 && adBlocks.map((block, index) => {
                                  const left = Math.max(0, (block.start / durationSeconds) * 100);
                                  const width = Math.max(0.5, ((block.end - block.start) / durationSeconds) * 100);
                                  return (
                                    <div
                                      key={`${block.start}-${block.end}-${index}`}
                                      className="absolute top-0 h-full rounded-full bg-rose-500/70"
                                      style={{ left: `${left}%`, width: `${width}%` }}
                                    />
                                  );
                                })}
                              </div>

                              <div className="flex justify-between text-xs text-gray-500">
                                {timelineTicks.map((tick) => (
                                  <span key={tick}>{formatTimelineLabel(durationSeconds * tick)}</span>
                                ))}
                              </div>

                              <div className="flex items-center gap-4 text-xs text-gray-500">
                                <span className="flex items-center gap-2">
                                  <span className="h-2 w-2 rounded-full bg-blue-500" />
                                  Content
                                </span>
                                <span className="flex items-center gap-2">
                                  <span className="h-2 w-2 rounded-full bg-rose-500" />
                                  Ads removed
                                </span>
                              </div>
                            </div>
                          </div>
                        );
                      })()}

                      <div>
                        <h3 className="font-semibold text-gray-900 mb-4 text-left">AI Model Performance</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-white border rounded-lg p-4">
                            <h4 className="font-medium text-gray-900 mb-3 text-left">Processing Status</h4>
                            <div className="space-y-2">
                              {Object.entries(stats.processing_stats?.model_call_statuses || {}).map(([status, count]) => (
                                <div key={status} className="flex justify-between items-center">
                                  <span className="text-sm text-gray-600 capitalize">{status}</span>
                                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                                    status === 'success' ? 'bg-green-100 text-green-800' :
                                    status === 'failed' ? 'bg-red-100 text-red-800' :
                                    'bg-gray-100 text-gray-800'
                                  }`}>
                                    {count}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>

                          <div className="bg-white border rounded-lg p-4">
                            <h4 className="font-medium text-gray-900 mb-3 text-left">Models Used</h4>
                            <div className="space-y-2">
                              {Object.entries(stats.processing_stats?.model_types || {}).map(([model, count]) => (
                                <div key={model} className="flex justify-between items-center">
                                  <span className="text-sm text-gray-600">{model}</span>
                                  <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                                    {count} calls
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === 'model-calls' && (
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-4 text-left">Model Calls ({stats.model_calls?.length || 0})</h3>
                      <div className="bg-white border rounded-lg overflow-hidden">
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                              <tr>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Segment Range</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Retries</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                              </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                              {(stats.model_calls || []).map((call) => (
                                <>
                                  <tr key={call.id} className="hover:bg-gray-50">
                                    <td className="px-4 py-3 text-sm text-gray-900">{call.id}</td>
                                    <td className="px-4 py-3 text-sm text-gray-900">{call.model_name}</td>
                                    <td className="px-4 py-3 text-sm text-gray-600">{call.segment_range}</td>
                                    <td className="px-4 py-3">
                                      <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                                        call.status === 'success' ? 'bg-green-100 text-green-800' :
                                        call.status === 'failed' ? 'bg-red-100 text-red-800' :
                                        'bg-yellow-100 text-yellow-800'
                                      }`}>
                                        {call.status}
                                      </span>
                                    </td>
                                    <td className="px-4 py-3 text-sm text-gray-600">{call.timestamp ? formatDateTime(call.timestamp) : 'N/A'}</td>
                                    <td className="px-4 py-3 text-sm text-gray-600">{call.retry_attempts}</td>
                                    <td className="px-4 py-3">
                                      <button
                                        onClick={() => toggleModelCallDetails(call.id)}
                                        className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                                      >
                                        {expandedModelCalls.has(call.id) ? 'Hide' : 'Details'}
                                      </button>
                                    </td>
                                  </tr>
                                  {expandedModelCalls.has(call.id) && (
                                    <tr className="bg-gray-50">
                                      <td colSpan={7} className="px-4 py-4">
                                        <div className="space-y-4">
                                          {call.prompt && (
                                            <div>
                                              <h5 className="font-medium text-gray-900 mb-2 text-left">Prompt:</h5>
                                              <div className="bg-gray-100 p-3 rounded text-sm font-mono whitespace-pre-wrap max-h-40 overflow-y-auto text-left">
                                                {call.prompt}
                                              </div>
                                            </div>
                                          )}
                                          {call.error_message && (
                                            <div>
                                              <h5 className="font-medium text-red-900 mb-2 text-left">Error Message:</h5>
                                              <div className="bg-red-50 p-3 rounded text-sm font-mono whitespace-pre-wrap text-left">
                                                {call.error_message}
                                              </div>
                                            </div>
                                          )}
                                          {call.response && (
                                            <div>
                                              <h5 className="font-medium text-gray-900 mb-2 text-left">Response:</h5>
                                              <div className="bg-gray-100 p-3 rounded text-sm font-mono whitespace-pre-wrap max-h-40 overflow-y-auto text-left">
                                                {call.response}
                                              </div>
                                            </div>
                                          )}
                                        </div>
                                      </td>
                                    </tr>
                                  )}
                                </>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === 'transcript' && (
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-4 text-left">Transcript Segments ({stats.transcript_segments?.length || 0})</h3>
                      <div className="bg-white border rounded-lg overflow-hidden">
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                              <tr>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Seq #</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time Range</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Speaker</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Label</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Text</th>
                              </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                              {mergedTranscriptRows.map((row, idx) => {
                                if (row.type === 'audio_marker') {
                                  const aseg = row.data;
                                  const duration = (aseg.end_time - aseg.start_time).toFixed(1);
                                  return (
                                    <tr key={`audio-${aseg.id}-${idx}`} className={getAudioMarkerBg(aseg.label)}>
                                      <td colSpan={6} className="px-4 py-1.5 text-center">
                                        <span className={`text-xs font-medium ${getAudioMarkerTextColor(aseg.label)}`}>
                                          [{getAudioMarkerLabel(aseg.label)}] ({duration}s)
                                        </span>
                                      </td>
                                    </tr>
                                  );
                                }
                                const segment = row.data;
                                return (
                                  <tr key={segment.id} className={`hover:bg-gray-50 ${
                                    segment.primary_label === 'ad' ? 'bg-red-50' : ''
                                  }`}>
                                    <td className="px-4 py-3 text-sm text-gray-900">{segment.sequence_num}</td>
                                    <td className="px-4 py-3 text-sm text-gray-600">
                                      {segment.start_time}s - {segment.end_time}s
                                    </td>
                                    <td className="px-4 py-3 text-sm">
                                      {segment.speaker ? (
                                        <span
                                          className="inline-flex px-2 py-0.5 rounded text-xs font-medium"
                                          style={{ backgroundColor: getSpeakerColor(segment.speaker) }}
                                        >
                                          {segment.speaker}
                                        </span>
                                      ) : null}
                                    </td>
                                    <td className="px-4 py-3">
                                      <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                                        segment.primary_label === 'ad'
                                          ? 'bg-red-100 text-red-800'
                                          : 'bg-green-100 text-green-800'
                                      }`}>
                                        {segment.primary_label === 'ad'
                                          ? (segment.mixed ? 'Ad (mixed)' : 'Ad')
                                          : 'Content'}
                                      </span>
                                    </td>
                                    <td className="px-4 py-3 text-sm text-gray-600">
                                      {formatConfidence(getTranscriptSegmentConfidence(segment))}
                                    </td>
                                    <td className="px-4 py-3 text-sm text-gray-900 max-w-md">
                                      <div className="truncate text-left" title={segment.text}>
                                        {segment.text}
                                      </div>
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === 'identifications' && (
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-4 text-left">Identifications ({stats.identifications?.length || 0})</h3>
                      <div className="bg-white border rounded-lg overflow-hidden">
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                              <tr>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Segment ID</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time Range</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Speaker</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Label</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model Call</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Text</th>
                              </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                              {mergedIdentificationRows.map((row, idx) => {
                                if (row.type === 'audio_marker') {
                                  const aseg = row.data;
                                  const duration = (aseg.end_time - aseg.start_time).toFixed(1);
                                  return (
                                    <tr key={`audio-${aseg.id}-${idx}`} className={getAudioMarkerBg(aseg.label)}>
                                      <td colSpan={8} className="px-4 py-1.5 text-center">
                                        <span className={`text-xs font-medium ${getAudioMarkerTextColor(aseg.label)}`}>
                                          [{getAudioMarkerLabel(aseg.label)}] ({duration}s)
                                        </span>
                                      </td>
                                    </tr>
                                  );
                                }
                                const identification = row.data;
                                return (
                                  <tr key={identification.id} className={`hover:bg-gray-50 ${
                                    identification.label === 'ad' ? 'bg-red-50' : ''
                                  }`}>
                                    <td className="px-4 py-3 text-sm text-gray-900">{identification.id}</td>
                                    <td className="px-4 py-3 text-sm text-gray-600">{identification.transcript_segment_id}</td>
                                    <td className="px-4 py-3 text-sm text-gray-600">
                                      {identification.segment_start_time}s - {identification.segment_end_time}s
                                    </td>
                                    <td className="px-4 py-3 text-sm">
                                      {identification.segment_speaker ? (
                                        <span
                                          className="inline-flex px-2 py-0.5 rounded text-xs font-medium"
                                          style={{ backgroundColor: getSpeakerColor(identification.segment_speaker) }}
                                        >
                                          {identification.segment_speaker}
                                        </span>
                                      ) : null}
                                    </td>
                                    <td className="px-4 py-3">
                                      <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                                        identification.label === 'ad'
                                          ? 'bg-red-100 text-red-800'
                                          : 'bg-green-100 text-green-800'
                                      }`}>
                                        {identification.label === 'ad'
                                          ? (identification.mixed ? 'ad (mixed)' : 'ad')
                                          : identification.label}
                                      </span>
                                    </td>
                                    <td className="px-4 py-3 text-sm text-gray-600">
                                      {formatConfidence(identification.confidence)}
                                    </td>
                                    <td className="px-4 py-3 text-sm text-gray-600">{identification.model_call_id}</td>
                                    <td className="px-4 py-3 text-sm text-gray-900 max-w-md">
                                      <div className="truncate text-left" title={identification.segment_text}>
                                        {identification.segment_text}
                                      </div>
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === 'audio' && hasAudioSegments && (
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-4 text-left">Audio Segments ({stats.audio_segments?.length || 0})</h3>
                      <div className="bg-white border rounded-lg overflow-hidden">
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                              <tr>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Duration</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Label</th>
                              </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                              {(stats.audio_segments || []).map((aseg) => {
                                const style = getAudioLabelStyle(aseg.label);
                                return (
                                  <tr key={aseg.id} className="hover:bg-gray-50">
                                    <td className="px-4 py-3 text-sm text-gray-600">
                                      {aseg.start_time}s - {aseg.end_time}s
                                    </td>
                                    <td className="px-4 py-3 text-sm text-gray-600">
                                      {formatDuration(aseg.end_time - aseg.start_time)}
                                    </td>
                                    <td className="px-4 py-3">
                                      <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${style.bg} ${style.text}`}>
                                        {aseg.label}
                                      </span>
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === 'speakers' && hasSpeakers && (
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-4 text-left">Speaker Statistics ({speakerStats.length})</h3>
                      <div className="bg-white border rounded-lg overflow-hidden">
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                              <tr>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Speaker</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Segments</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Words</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">% of Total</th>
                              </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                              {speakerStats.map((spk) => (
                                <tr key={spk.speaker} className="hover:bg-gray-50">
                                  <td className="px-4 py-3 text-sm">
                                    <span
                                      className="inline-flex px-2 py-0.5 rounded text-xs font-medium"
                                      style={{ backgroundColor: getSpeakerColor(spk.speaker) }}
                                    >
                                      {spk.speaker}
                                    </span>
                                  </td>
                                  <td className="px-4 py-3 text-sm text-gray-900">{spk.segmentCount}</td>
                                  <td className="px-4 py-3 text-sm text-gray-900">{formatDuration(spk.totalTime)}</td>
                                  <td className="px-4 py-3 text-sm text-gray-900">{spk.wordCount.toLocaleString()}</td>
                                  <td className="px-4 py-3 text-sm">
                                    <div className="flex items-center gap-2">
                                      <div className="flex-1 bg-gray-200 rounded-full h-2 max-w-[120px]">
                                        <div
                                          className="bg-blue-500 h-2 rounded-full"
                                          style={{ width: `${Math.min(100, spk.percentOfTotal)}%` }}
                                        />
                                      </div>
                                      <span className="text-gray-600">{spk.percentOfTotal.toFixed(1)}%</span>
                                    </div>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              ) : null}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
