"""migration

Revision ID: 3e5eebc6b3b1
Revises: 7033e6931510
Create Date: 2026-03-15 10:44:20.624296

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "3e5eebc6b3b1"
down_revision = "7033e6931510"
branch_labels = None
depends_on = None


def column_exists(table_name: str, column_name: str) -> bool:
    conn = op.get_bind()
    result = conn.execute(sa.text(f"PRAGMA table_info({table_name})"))
    columns = [row[1] for row in result.fetchall()]
    return column_name in columns


def upgrade():
    if not column_exists("feed", "enable_llm_chapter_fallback_tagging"):
        with op.batch_alter_table("feed", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column(
                    "enable_llm_chapter_fallback_tagging", sa.Boolean(), nullable=True
                )
            )


def downgrade():
    if column_exists("feed", "enable_llm_chapter_fallback_tagging"):
        with op.batch_alter_table("feed", schema=None) as batch_op:
            batch_op.drop_column("enable_llm_chapter_fallback_tagging")
