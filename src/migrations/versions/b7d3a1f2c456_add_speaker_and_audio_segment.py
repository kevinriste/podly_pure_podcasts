"""add speaker column and audio_segment table

Revision ID: b7d3a1f2c456
Revises: 4f9b2a6c8e11
Create Date: 2026-04-02 01:30:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "b7d3a1f2c456"
down_revision = "4f9b2a6c8e11"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("transcript_segment") as batch_op:
        batch_op.add_column(sa.Column("speaker", sa.String(50), nullable=True))

    op.create_table(
        "audio_segment",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("post_id", sa.Integer(), sa.ForeignKey("post.id"), nullable=False),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("label", sa.String(20), nullable=False),
        sa.Column("model_call_id", sa.Integer(), sa.ForeignKey("model_call.id"), nullable=True),
    )
    op.create_index("ix_audio_segment_post_id", "audio_segment", ["post_id"])


def downgrade() -> None:
    op.drop_index("ix_audio_segment_post_id", table_name="audio_segment")
    op.drop_table("audio_segment")

    with op.batch_alter_table("transcript_segment") as batch_op:
        batch_op.drop_column("speaker")
