"""Change feed ad_detection_strategy default from 'llm' to 'inherit'

Revision ID: cdd15af70147
Revises: 16724c87ef43
Create Date: 2026-02-13 22:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "cdd15af70147"
down_revision = "16724c87ef43"
branch_labels = None
depends_on = None


def upgrade():
    # Convert existing "llm" values to "inherit" (use user default).
    # Keep "chapter" rows as-is since they represent a feed-specific capability.
    op.execute(
        "UPDATE feed SET ad_detection_strategy = 'inherit'"
        " WHERE ad_detection_strategy = 'llm'"
    )


def downgrade():
    op.execute(
        "UPDATE feed SET ad_detection_strategy = 'llm'"
        " WHERE ad_detection_strategy = 'inherit'"
    )
