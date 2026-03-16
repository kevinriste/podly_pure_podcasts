import { toast } from 'react-hot-toast';
import { configApi } from '../../../services/api';
import { useConfigContext } from '../ConfigContext';
import { Section, Field, SaveButton, TestButton } from '../shared';
import type { LLMConfig } from '../../../types';

export default function OneShotSection() {
  const { pending, setField, getEnvHint, handleSave, isSaving } = useConfigContext();

  if (!pending) return null;

  const oneshotEnvMeta = getEnvHint('llm.oneshot_model');
  const oneshotLockedByEnv = oneshotEnvMeta?.value !== undefined;

  const handleTestOneShot = () => {
    toast.promise(configApi.testOneShot({ llm: pending.llm as LLMConfig }), {
      loading: 'Testing One-shot connection...',
      success: (res: { ok: boolean; message?: string }) =>
        res?.message || 'One-shot connection OK',
      error: (err: unknown) => {
        const e = err as {
          response?: { data?: { error?: string; message?: string } };
          message?: string;
        };
        return (
          e?.response?.data?.error ||
          e?.response?.data?.message ||
          e?.message ||
          'One-shot connection failed'
        );
      },
    });
  };

  return (
    <div className="space-y-6">
      <Section title="One-shot">
        <div className="mb-3 rounded border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">
          One-shot API key lookup order: <code>ONESHOT_API_KEY</code>, then{' '}
          <code>LLM_API_KEY</code>, then the saved <code>LLM API Key</code> value.
          The Test One-shot button uses this same order.
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <Field label="One-shot Model" envMeta={oneshotEnvMeta}>
            <input
              className="input"
              type="text"
              value={pending?.llm?.oneshot_model ?? ''}
              onChange={(e) =>
                setField(
                  ['llm', 'oneshot_model'],
                  e.target.value.trim() === '' ? null : e.target.value
                )
              }
              disabled={oneshotLockedByEnv}
              placeholder="e.g. openai/gpt-5-mini"
            />
            <p className="text-xs text-gray-500 mt-1">
              {oneshotLockedByEnv
                ? 'Set via environment variable; editing is disabled.'
                : 'Used when ad detection strategy is set to One-shot LLM.'}
            </p>
          </Field>

          <Field label="One-shot Max Chunk Duration (sec)">
            <input
              className="input"
              type="number"
              min={1}
              value={pending?.llm?.oneshot_max_chunk_duration_seconds ?? 7200}
              onChange={(e) =>
                setField(
                  ['llm', 'oneshot_max_chunk_duration_seconds'],
                  Number(e.target.value)
                )
              }
            />
          </Field>

          <Field label="One-shot Chunk Overlap (sec)">
            <input
              className="input"
              type="number"
              min={0}
              value={pending?.llm?.oneshot_chunk_overlap_seconds ?? 900}
              onChange={(e) =>
                setField(['llm', 'oneshot_chunk_overlap_seconds'], Number(e.target.value))
              }
            />
          </Field>
        </div>

        <TestButton onClick={handleTestOneShot} label="Test One-shot" />
      </Section>

      <SaveButton onSave={handleSave} isPending={isSaving} />

      <style>{`.input{width:100%;padding:0.5rem;border:1px solid #e5e7eb;border-radius:0.375rem;font-size:0.875rem}`}</style>
    </div>
  );
}
