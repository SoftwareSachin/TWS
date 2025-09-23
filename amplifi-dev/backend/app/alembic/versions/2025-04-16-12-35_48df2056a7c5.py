"""empty message

Revision ID: 48df2056a7c5
Revises: f0c49b63a063, 929a3548941f
Create Date: 2025-04-16 12:35:52.902264

"""
from alembic import op
import sqlalchemy as sa
import sqlalchemy_utils
import sqlmodel # added


# revision identifiers, used by Alembic.
revision = '48df2056a7c5'
down_revision = ('f0c49b63a063', '929a3548941f')
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm") 
    pass


def downgrade():
    pass
