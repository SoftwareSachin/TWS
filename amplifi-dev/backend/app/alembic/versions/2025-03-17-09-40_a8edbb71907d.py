"""empty message

Revision ID: a8edbb71907d
Revises: e1bf06f3cc65, 3bfb31e6514e
Create Date: 2025-03-17 09:40:04.908123

"""
from alembic import op
import sqlalchemy as sa
import sqlalchemy_utils
import sqlmodel # added


# revision identifiers, used by Alembic.
revision = 'a8edbb71907d'
down_revision = ('e1bf06f3cc65', '3bfb31e6514e')
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm") 
    pass


def downgrade():
    pass
