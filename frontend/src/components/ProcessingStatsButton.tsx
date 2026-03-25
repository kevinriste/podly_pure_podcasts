import ChapterProcessingStats from './ChapterProcessingStats';
import LLMProcessingStats from './LLMProcessingStats';

interface ProcessingStatsButtonProps {
  episodeGuid: string;
  hasProcessedAudio: boolean;
  adDetectionStrategy?: 'inherit' | 'llm' | 'oneshot' | 'chapter' | 'chapter_insert';
  className?: string;
}

export default function ProcessingStatsButton({
  episodeGuid,
  hasProcessedAudio,
  adDetectionStrategy = 'llm',
  className = ''
}: ProcessingStatsButtonProps) {
  if (!hasProcessedAudio) {
    return null;
  }

  if (
    adDetectionStrategy === 'chapter' ||
    adDetectionStrategy === 'chapter_insert'
  ) {
    return <ChapterProcessingStats episodeGuid={episodeGuid} isStatsReady={hasProcessedAudio} className={className} />;
  }

  return <LLMProcessingStats episodeGuid={episodeGuid} isStatsReady={hasProcessedAudio} className={className} />;
}
