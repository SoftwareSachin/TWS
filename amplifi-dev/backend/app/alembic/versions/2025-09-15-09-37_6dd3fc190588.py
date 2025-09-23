"""empty message

Revision ID: 6dd3fc190588
Revises: 29a594d18eb1
Create Date: 2025-08-14 09:37:12.441970

"""
from alembic import op
import sqlalchemy as sa
import sqlalchemy_utils
import sqlmodel # added


# revision identifiers, used by Alembic.
revision = '6dd3fc190588'
down_revision = 'd2e2399b0a95'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm") 
    
    # Add VideoSegment to the existing ChunkTypeEnum
    conn = op.get_bind()
    
    # Check if chunktypeenum exists
    result = conn.execute(sa.text("SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'chunktypeenum')")).scalar()
    if result:
        # Add VideoSegment enum value if it doesn't exist
        op.execute("ALTER TYPE chunktypeenum ADD VALUE IF NOT EXISTS 'VideoSegment'")
    else:
        # Create the enum type with all values if it doesn't exist
        op.execute(sa.text("""
            CREATE TYPE chunktypeenum AS ENUM (
                'ImageDescription', 'ImageText', 'ImageObject', 
                'AudioSegment', 'Speaker', 'VideoScene', 'VideoSegment',
                'PDFText', 'PDFTable'
            )
        """))
    
    # ### end Alembic commands ###


def downgrade():
    # Note: Removing enum values can be complex and may require data migration
    # For safety, we don't automatically remove the enum value
    pass
    # ### end Alembic commands ###
