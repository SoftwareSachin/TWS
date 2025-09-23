"""add GPT41 to ChatModelEnum

Revision ID: 929a3548941f
Revises: ec7ec955a781
Create Date: 2025-04-15 21:44:53.567691
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "929a3548941f"
down_revision = "ec7ec955a781"
branch_labels = None
depends_on = None


def upgrade():
    # Manually Written: Add GPT41 to the ChatModelEnum type
    op.execute("ALTER TYPE chatmodelenum ADD VALUE IF NOT EXISTS 'GPT41'")


def downgrade():
    # Can't downgrade since removing values from an enum type is not supported in PostgreSQL.
    # Shouldn't be big issue.
    pass
