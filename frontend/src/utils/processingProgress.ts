export type ProcessingPipelineVariant =
  | 'llm'
  | 'chapter'
  | 'chapter_insert'
  | 'generic';

export interface ProcessingProgressInput {
  status?: string | null;
  step?: number | null;
  totalSteps?: number | null;
  stepName?: string | null;
  progressPercentage?: number | null;
}

export interface ProcessingStageState {
  index: number;
  label: string;
  shortLabel: string;
  state: 'pending' | 'active' | 'completed' | 'failed';
}

export interface ProcessingProgressModel {
  variant: ProcessingPipelineVariant;
  progress: number;
  currentStep: number;
  currentStageLabel: string;
  stages: ProcessingStageState[];
}

interface StageDefinition {
  label: string;
  shortLabel: string;
}

const LLM_STAGES: StageDefinition[] = [
  { label: 'Queued', shortLabel: 'Queue' },
  { label: 'Downloading episode', shortLabel: 'Download' },
  { label: 'Transcribing audio', shortLabel: 'Transcribe' },
  { label: 'Identifying ads', shortLabel: 'Identify' },
  { label: 'Processing audio', shortLabel: 'Process' },
];

const CHAPTER_STAGES: StageDefinition[] = [
  { label: 'Queued', shortLabel: 'Queue' },
  { label: 'Downloading episode', shortLabel: 'Download' },
  { label: 'Reading chapters', shortLabel: 'Read' },
  { label: 'Chapters filtered', shortLabel: 'Filter' },
  { label: 'Processing audio', shortLabel: 'Process' },
];

const CHAPTER_INSERT_STAGES: StageDefinition[] = [
  { label: 'Queued', shortLabel: 'Queue' },
  { label: 'Downloading episode', shortLabel: 'Download' },
  { label: 'Resolving chapters', shortLabel: 'Resolve' },
  { label: 'Generating chapters', shortLabel: 'Generate' },
  { label: 'Copying audio and writing chapters', shortLabel: 'Write' },
];

const GENERIC_STAGES: StageDefinition[] = [
  { label: 'Queued', shortLabel: 'Queue' },
  { label: 'Downloading episode', shortLabel: 'Download' },
  { label: 'Transcript or chapter stage', shortLabel: 'Stage 2' },
  { label: 'Classification or chapter stage', shortLabel: 'Stage 3' },
  { label: 'Audio processing or chapter write', shortLabel: 'Stage 4' },
];

const PIPELINE_MAP: Record<ProcessingPipelineVariant, StageDefinition[]> = {
  llm: LLM_STAGES,
  chapter: CHAPTER_STAGES,
  chapter_insert: CHAPTER_INSERT_STAGES,
  generic: GENERIC_STAGES,
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function normalizeStep(step: number | null | undefined): number {
  if (typeof step !== 'number' || !Number.isFinite(step)) {
    return 0;
  }
  return clamp(Math.round(step), 0, 4);
}

function normalizeProgress(
  status: string,
  step: number,
  totalSteps: number,
  progressPercentage: number | null | undefined
): number {
  if (
    typeof progressPercentage === 'number' &&
    Number.isFinite(progressPercentage)
  ) {
    return clamp(progressPercentage, 0, 100);
  }

  if (status === 'completed' || status === 'skipped') {
    return 100;
  }

  const safeTotal = totalSteps > 0 ? totalSteps : 4;
  return clamp((step / safeTotal) * 100, 0, 100);
}

function inferPipelineVariant(
  stepName: string | null | undefined
): ProcessingPipelineVariant {
  const raw = (stepName || '').toLowerCase();
  if (!raw) {
    return 'generic';
  }

  if (
    raw.includes('resolving chapters') ||
    raw.includes('chapter generation') ||
    raw.includes('chapters resolved') ||
    raw.includes('copying audio and writing chapters')
  ) {
    return 'chapter_insert';
  }

  if (raw.includes('reading chapters') || raw.includes('chapters filtered')) {
    return 'chapter';
  }

  if (raw.includes('transcribing audio') || raw.includes('identifying ads')) {
    return 'llm';
  }

  return 'generic';
}

function currentStageLabel(
  stepName: string | null | undefined,
  stages: StageDefinition[],
  currentStep: number
): string {
  const label = (stepName || '').trim();
  if (!label) {
    return stages[currentStep]?.label ?? 'Queued';
  }

  // Avoid overly verbose queue text in compact cards.
  if (label.toLowerCase().startsWith('queued for processing')) {
    return 'Queued for processing';
  }

  return label;
}

export function buildProcessingProgressModel(
  input: ProcessingProgressInput
): ProcessingProgressModel {
  const status = (input.status || '').toLowerCase();
  const currentStep = normalizeStep(input.step);
  const totalSteps =
    typeof input.totalSteps === 'number' && Number.isFinite(input.totalSteps)
      ? input.totalSteps
      : 4;
  const variant = inferPipelineVariant(input.stepName);
  const stages = PIPELINE_MAP[variant];

  const completedAll = status === 'completed' || status === 'skipped';
  const failedLike = status === 'failed' || status === 'cancelled' || status === 'error';

  const stageStates: ProcessingStageState[] = stages.map((stage, index) => {
    let state: ProcessingStageState['state'] = 'pending';
    if (completedAll) {
      state = 'completed';
    } else if (failedLike && index === currentStep) {
      state = 'failed';
    } else if (index < currentStep) {
      state = 'completed';
    } else if (
      index === currentStep &&
      (status === 'running' || status === 'pending' || status === 'starting' || status === 'processing')
    ) {
      state = 'active';
    }

    return {
      index,
      label: stage.label,
      shortLabel: stage.shortLabel,
      state,
    };
  });

  return {
    variant,
    progress: normalizeProgress(
      status,
      currentStep,
      totalSteps,
      input.progressPercentage
    ),
    currentStep,
    currentStageLabel: currentStageLabel(input.stepName, stages, currentStep),
    stages: stageStates,
  };
}

