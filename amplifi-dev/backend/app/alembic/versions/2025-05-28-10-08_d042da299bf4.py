"""empty message

Revision ID: d042da299bf4
Revises: e22028eb53f9
Create Date: 2025-05-28 10:08:53.145069

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'd042da299bf4'
down_revision = 'e22028eb53f9'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    conn = op.get_bind()

    # Create enum type tooltype if not exists
    if not conn.execute(text("SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'tooltype')")).scalar():
        op.execute(text("CREATE TYPE tooltype AS ENUM ('system', 'mcp')"))

    # Create enum type mcptype if not exists
    if not conn.execute(text("SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'mcptype')")).scalar():
        op.execute(text("CREATE TYPE mcptype AS ENUM ('internal', 'external')"))

    # Create tools table without tool_kind column first
    op.create_table(
        'tools',
        sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('description', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('deprecated', sa.Boolean(), nullable=False),
        sa.Column('tool_metadata', sa.JSON(), nullable=True),
        sa.Column('id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_tools_id'), 'tools', ['id'], unique=False)
    op.create_index(op.f('ix_tools_name'), 'tools', ['name'], unique=False)

    # Add the enum column tool_kind separately
    op.execute(text("ALTER TABLE tools ADD COLUMN tool_kind tooltype NOT NULL"))

    # Create mcp_tools table without mcp_subtype first
    op.create_table(
        'mcp_tools',
        sa.Column('mcp_server_config', sa.JSON(), nullable=True),
        sa.Column('timeout_secs', sa.Integer(), nullable=False),
        sa.Column('description', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('tool_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
        sa.ForeignKeyConstraint(['tool_id'], ['tools.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tool_id'),
    )
    op.create_index(op.f('ix_mcp_tools_id'), 'mcp_tools', ['id'], unique=False)

    # Add the enum column mcp_subtype separately
    op.execute(text("ALTER TABLE mcp_tools ADD COLUMN mcp_subtype mcptype NOT NULL"))

    # Create system_tools table (no enums here)
    op.create_table(
        'system_tools',
        sa.Column('python_module', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('function_name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('is_async', sa.Boolean(), nullable=False),
        sa.Column('input_schema', sa.JSON(), nullable=True),
        sa.Column('output_schema', sa.JSON(), nullable=True),
        sa.Column('function_signature', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('tool_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
        sa.ForeignKeyConstraint(['tool_id'], ['tools.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tool_id'),
    )
    op.create_index(op.f('ix_system_tools_id'), 'system_tools', ['id'], unique=False)

    # Create workspace_tools table
    op.create_table(
        'workspace_tools',
        sa.Column('dataset_ids', postgresql.ARRAY(sa.UUID()), nullable=True),
        sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('mcp_tools', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('workspace_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
        sa.Column('tool_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
        sa.ForeignKeyConstraint(['tool_id'], ['tools.id']),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_workspace_tools_id'), 'workspace_tools', ['id'], unique=False)
    op.create_index(op.f('ix_workspace_tools_name'), 'workspace_tools', ['name'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_workspace_tools_name'), table_name='workspace_tools')
    op.drop_index(op.f('ix_workspace_tools_id'), table_name='workspace_tools')
    op.drop_table('workspace_tools')

    op.drop_index(op.f('ix_system_tools_id'), table_name='system_tools')
    op.drop_table('system_tools')

    op.drop_index(op.f('ix_mcp_tools_id'), table_name='mcp_tools')
    op.drop_table('mcp_tools')

    op.drop_index(op.f('ix_tools_name'), table_name='tools')
    op.drop_index(op.f('ix_tools_id'), table_name='tools')
    op.drop_table('tools')

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS mcptype")
    op.execute("DROP TYPE IF EXISTS tooltype")
