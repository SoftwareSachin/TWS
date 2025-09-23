"""Add new document types to DocumentTypeEnum

Revision ID: 90f175c9fce9
Revises: 96c82fbd95bc
Create Date: 2025-06-24 07:01:30.105468

"""
from alembic import op
import sqlalchemy as sa
import sqlalchemy_utils
import sqlmodel # added
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '90f175c9fce9'
down_revision = '96c82fbd95bc'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm") 
    
    # Add new document types to the existing DocumentTypeEnum
    conn = op.get_bind()
    
    # Check if documenttypeenum exists
    result = conn.execute(sa.text("SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'documenttypeenum')")).scalar()
    if result:
        # Add new enum values if they don't exist
        op.execute("ALTER TYPE documenttypeenum ADD VALUE IF NOT EXISTS 'Markdown'")
        op.execute("ALTER TYPE documenttypeenum ADD VALUE IF NOT EXISTS 'HTML'")
        op.execute("ALTER TYPE documenttypeenum ADD VALUE IF NOT EXISTS 'CSV'")
        op.execute("ALTER TYPE documenttypeenum ADD VALUE IF NOT EXISTS 'XLSX'")
        op.execute("ALTER TYPE documenttypeenum ADD VALUE IF NOT EXISTS 'PPTX'")
        op.execute("ALTER TYPE documenttypeenum ADD VALUE IF NOT EXISTS 'DOCX'")
    else:
        # Create the enum type with all values if it doesn't exist
        op.execute(sa.text("""
            CREATE TYPE documenttypeenum AS ENUM (
                'Image', 'Audio', 'Video', 'PDF', 
                'Markdown', 'HTML', 'CSV', 'XLSX', 'PPTX', 'DOCX'
            )
        """))

    # ### end Alembic commands ###


def downgrade():

    pass # Downgrade logic is not implemented, as removing enum values can be complex and may require data migration.
    # ### end Alembic commands ###