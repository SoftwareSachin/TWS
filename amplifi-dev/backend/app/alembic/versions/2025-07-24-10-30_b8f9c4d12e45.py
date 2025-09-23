"""Update dataset_required for Vector Search , Text to SQL tools and Visualization Tool

Revision ID: b8f9c4d12e45
Revises: 8d12bfd5a570
Create Date: 2025-07-24 10:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'b8f9c4d12e45'
down_revision = '8d12bfd5a570'
branch_labels = None
depends_on = None


def upgrade():
    """Update dataset_required to True for tools that require datasets."""
    conn = op.get_bind()
    
    # Tools that require datasets
    tools_requiring_datasets = [
        "Vector Search Tool",
        "Text to SQL Tool",
        "Visualization Tool",
    ]
    
    # Update dataset_required to True for specified tools
    conn.execute(
        sa.text("""
            UPDATE tools 
            SET dataset_required = true 
            WHERE name = ANY(:tool_names) AND dataset_required = false
        """),
        {"tool_names": tools_requiring_datasets}
    )


def downgrade():
    """Revert dataset_required to False for the updated tools."""
    conn = op.get_bind()
    
    # Tools that were updated (same list as upgrade)
    tools_requiring_datasets = [
        "Vector Search Tool",
        "Text to SQL Tool",
        "Visualization Tool",
    ]
    
    # Revert dataset_required to False for specified tools
    conn.execute(
        sa.text("""
            UPDATE tools 
            SET dataset_required = false 
            WHERE name = ANY(:tool_names) AND dataset_required = true
        """),
        {"tool_names": tools_requiring_datasets}
    )
