"""empty message

Revision ID: c78d3f1ca536
Revises: db7db49a245a
Create Date: 2025-07-08 05:54:05.535856

"""
from alembic import op
import sqlalchemy as sa
import sqlalchemy_utils
import sqlmodel  # added
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'c78d3f1ca536'
down_revision = 'db7db49a245a'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm") 
    
    # Add the new column
    op.add_column(
        'tools',
        sa.Column('organization_id', sqlmodel.sql.sqltypes.GUID(), nullable=True)
    )
    
    # Explicitly name the foreign key constraint
    op.create_foreign_key(
        'tools_organization_id_fkey',  # Named constraint
        'tools',
        'organizations',
        ['organization_id'],
        ['id']
    )


def downgrade():
    # Drop the foreign key constraint using the explicit name
    op.drop_constraint('tools_organization_id_fkey', 'tools', type_='foreignkey')
    
    # Drop the column
    op.drop_column('tools', 'organization_id')
