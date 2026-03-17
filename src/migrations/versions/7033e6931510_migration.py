"""migration

Revision ID: 7033e6931510
Revises: 4fbeddeb6a6c
Create Date: 2026-02-22 21:16:17.360249

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "7033e6931510"
down_revision = "4fbeddeb6a6c"
branch_labels = None
depends_on = None


def column_exists(table_name: str, column_name: str) -> bool:
    conn = op.get_bind()
    result = conn.execute(sa.text(f"PRAGMA table_info({table_name})"))
    columns = [row[1] for row in result.fetchall()]
    return column_name in columns


def upgrade():
    if not column_exists("llm_settings", "enable_llm_chapter_fallback_tagging"):
        with op.batch_alter_table("llm_settings", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column(
                    "enable_llm_chapter_fallback_tagging",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.false(),
                )
            )


def downgrade():
    if column_exists("llm_settings", "enable_llm_chapter_fallback_tagging"):
        with op.batch_alter_table("llm_settings", schema=None) as batch_op:
            batch_op.drop_column("enable_llm_chapter_fallback_tagging")
