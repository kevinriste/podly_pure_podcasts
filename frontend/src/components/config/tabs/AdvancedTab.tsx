import { useConfigContext, type AdvancedSubtab } from '../ConfigContext';
import {
  LLMSection,
  WhisperSection,
  ProcessingSection,
  OutputSection,
  AppSection,
} from '../sections';

const SUBTABS: { id: AdvancedSubtab; label: string }[] = [
  { id: 'llm', label: 'LLM' },
  { id: 'whisper', label: 'Whisper' },
  { id: 'processing', label: 'Processing' },
  { id: 'output', label: 'Output' },
  { id: 'app', label: 'App' },
];

export default function AdvancedTab() {
  const { activeSubtab, setActiveSubtab } = useConfigContext();

  return (
    <div className="space-y-6">
      {/* Subtab Navigation */}
      <div className="flex space-x-2 flex-wrap gap-y-2">
        {SUBTABS.map((subtab) => (
          <button
            key={subtab.id}
            onClick={() => setActiveSubtab(subtab.id)}
            className={`px-3 py-1.5 text-sm rounded-md font-medium transition-colors ${
              activeSubtab === subtab.id
                ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-500/20 dark:text-indigo-100 dark:ring-1 dark:ring-indigo-400/40'
                : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-slate-100'
            }`}
          >
            {subtab.label}
          </button>
        ))}
      </div>

      {/* Subtab Content */}
      <div>
        {activeSubtab === 'llm' && <LLMSection />}
        {activeSubtab === 'whisper' && <WhisperSection />}
        {activeSubtab === 'processing' && <ProcessingSection />}
        {activeSubtab === 'output' && <OutputSection />}
        {activeSubtab === 'app' && <AppSection />}
      </div>
    </div>
  );
}
