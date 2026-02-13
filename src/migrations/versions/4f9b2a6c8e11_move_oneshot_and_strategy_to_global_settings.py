"""Move oneshot model and ad strategy to global settings

Revision ID: 4f9b2a6c8e11
Revises: cdd15af70147
Create Date: 2026-02-13 17:15:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "4f9b2a6c8e11"
down_revision = "cdd15af70147"
branch_labels = None
depends_on = None


def upgrade():
    # Add new global settings columns.
    with op.batch_alter_table("llm_settings", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("oneshot_model", sa.String(length=100), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "oneshot_max_chunk_duration_seconds",
                sa.Integer(),
                nullable=False,
                server_default="7200",
            )
        )
        batch_op.add_column(
            sa.Column(
                "oneshot_chunk_overlap_seconds",
                sa.Integer(),
                nullable=False,
                server_default="900",
            )
        )

    with op.batch_alter_table("app_settings", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "ad_detection_strategy",
                sa.String(length=20),
                nullable=False,
                server_default="llm",
            )
        )

    conn = op.get_bind()

    # Carry over the first explicit non-default user strategy, if present.
    user_strategy = conn.execute(
        sa.text(
            "SELECT ad_detection_strategy FROM users "
            "WHERE ad_detection_strategy IS NOT NULL "
            "AND ad_detection_strategy != 'llm' "
            "ORDER BY id ASC LIMIT 1"
        )
    ).scalar()
    if isinstance(user_strategy, str) and user_strategy in {"llm", "oneshot"}:
        conn.execute(
            sa.text("UPDATE app_settings SET ad_detection_strategy = :strategy"),
            {"strategy": user_strategy},
        )

    # Carry over the first user-specific oneshot model override, if present.
    user_oneshot_model = conn.execute(
        sa.text(
            "SELECT oneshot_model FROM users "
            "WHERE oneshot_model IS NOT NULL AND oneshot_model != '' "
            "ORDER BY id ASC LIMIT 1"
        )
    ).scalar()
    if isinstance(user_oneshot_model, str) and user_oneshot_model.strip():
        conn.execute(
            sa.text("UPDATE llm_settings SET oneshot_model = :model"),
            {"model": user_oneshot_model.strip()},
        )

    # Remove per-user fields.
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_column("oneshot_model")
        batch_op.drop_column("ad_detection_strategy")


def downgrade():
    # Restore per-user fields.
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "ad_detection_strategy",
                sa.String(length=20),
                nullable=False,
                server_default="llm",
            )
        )
        batch_op.add_column(
            sa.Column("oneshot_model", sa.String(length=100), nullable=True)
        )

    conn = op.get_bind()

    # Restore app strategy to users.
    app_strategy = conn.execute(
        sa.text("SELECT ad_detection_strategy FROM app_settings LIMIT 1")
    ).scalar()
    if isinstance(app_strategy, str) and app_strategy in {"llm", "oneshot"}:
        conn.execute(
            sa.text("UPDATE users SET ad_detection_strategy = :strategy"),
            {"strategy": app_strategy},
        )

    # Restore global oneshot model to users.
    llm_oneshot_model = conn.execute(
        sa.text("SELECT oneshot_model FROM llm_settings LIMIT 1")
    ).scalar()
    if isinstance(llm_oneshot_model, str) and llm_oneshot_model.strip():
        conn.execute(
            sa.text("UPDATE users SET oneshot_model = :model"),
            {"model": llm_oneshot_model.strip()},
        )

    # Remove global fields.
    with op.batch_alter_table("app_settings", schema=None) as batch_op:
        batch_op.drop_column("ad_detection_strategy")

    with op.batch_alter_table("llm_settings", schema=None) as batch_op:
        batch_op.drop_column("oneshot_chunk_overlap_seconds")
        batch_op.drop_column("oneshot_max_chunk_duration_seconds")
        batch_op.drop_column("oneshot_model")
