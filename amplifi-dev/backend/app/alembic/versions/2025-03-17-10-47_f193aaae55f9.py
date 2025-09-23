"""empty message

Revision ID: f193aaae55f9
Revises: a8edbb71907d, d75d91ad2b53
Create Date: 2025-03-17 10:47:03.548461

"""
from alembic import op
import sqlalchemy as sa
import sqlalchemy_utils
import sqlmodel # added


# revision identifiers, used by Alembic.
revision = 'f193aaae55f9'
down_revision = ('a8edbb71907d', 'd75d91ad2b53')
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm") 
    pass


def downgrade():
    pass
