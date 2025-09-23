"""empty message

Revision ID: 96c82fbd95bc
Revises: d85c1a883bc0
Create Date: 2025-06-13 04:08:31.286505

"""
from alembic import op
import sqlalchemy as sa
import sqlalchemy_utils
import sqlmodel  # added


# revision identifiers, used by Alembic.
revision = '96c82fbd95bc'
down_revision = 'd85c1a883bc0'
branch_labels = None
depends_on = None


def upgrade():
    # Ensure pg_trgm extension exists
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # Manually Written: Add GPTo3 to the ChatModelEnum type
    op.execute("ALTER TYPE chatmodelenum ADD VALUE IF NOT EXISTS 'GPTo3'")

    # Explicit cast required when changing to ENUM type
    op.execute("""
        ALTER TABLE agents
        ALTER COLUMN llm_model TYPE chatmodelenum
        USING llm_model::chatmodelenum
    """)


def downgrade():
    # Revert column type back to VARCHAR explicitly
    op.execute("""
        ALTER TABLE agents
        ALTER COLUMN llm_model TYPE VARCHAR
        USING llm_model::text
    """)
