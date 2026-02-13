import { useQuery } from '@tanstack/react-query';
import { feedsApi } from '../services/api';
import ChapterProcessingStats from './ChapterProcessingStats';
import LLMProcessingStats from './LLMProcessingStats';
import { useEpisodeStatus } from '../hooks/useEpisodeStatus';

interface ProcessingStatsButtonProps {
  episodeGuid: string;
  isWhitelisted: boolean;
  hasProcessedAudio: boolean;
  feedId?: number;
  className?: string;
}

export default function ProcessingStatsButton({
  episodeGuid,
  isWhitelisted,
  hasProcessedAudio,
  feedId,
  className = ''
}: ProcessingStatsButtonProps) {
  const { data: status } = useEpisodeStatus(episodeGuid, isWhitelisted, hasProcessedAudio, feedId);
  const { data: stats } = useQuery({
    queryKey: ['episode-stats', episodeGuid],
    queryFn: () => feedsApi.getPostStats(episodeGuid),
    enabled: false,
    staleTime: 0,
  });

  const isInProcessingAudioPhase =
    !hasProcessedAudio &&
    status?.status === 'running' &&
    status.step_name?.trim().toLowerCase() === 'processing audio';
  const statsReady = hasProcessedAudio || isInProcessingAudioPhase;

  if (!statsReady) {
    return null;
  }

  if (stats?.ad_detection_strategy === 'chapter') {
    return <ChapterProcessingStats episodeGuid={episodeGuid} isStatsReady={statsReady} className={className} />;
  }

  return <LLMProcessingStats episodeGuid={episodeGuid} isStatsReady={statsReady} className={className} />;
}
