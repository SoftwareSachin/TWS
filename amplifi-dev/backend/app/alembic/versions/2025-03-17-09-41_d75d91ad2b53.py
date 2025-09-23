"""empty message

Revision ID: d75d91ad2b53
Revises: 433dceb5e9e5, 3bfb31e6514e
Create Date: 2025-03-17 09:41:45.985936

"""
from alembic import op
import sqlalchemy as sa
import sqlalchemy_utils
import sqlmodel # added


# revision identifiers, used by Alembic.
revision = 'd75d91ad2b53'
down_revision = ('433dceb5e9e5', '3bfb31e6514e')
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm") 
    pass


def downgrade():
    pass
