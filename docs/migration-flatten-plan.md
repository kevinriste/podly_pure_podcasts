# Migration Flatten Plan (Do Not Execute Yet)

1. Wait until this feature set is merged and no open branches depend on intermediate revisions.
2. Create a fresh baseline migration from current models (`alembic revision --autogenerate`) in a temporary branch.
3. Diff the generated baseline against the live schema in dev/staging to confirm parity.
4. Replace historical migration chain with:
   - one baseline schema migration
   - one data backfill migration (if still needed for legacy DB upgrades)
5. Test upgrade paths:
   - fresh DB -> latest head
   - last pre-flatten production revision -> latest head
6. Test downgrade one step for rollback safety.
7. Update release notes with an explicit "flattened migration history" section and required upgrade order.
8. Merge only after staging verification on a copy of production data.
