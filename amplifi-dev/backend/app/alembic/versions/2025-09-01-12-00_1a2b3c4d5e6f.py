"""Add Graph Search system tool

Revision ID: 1a2b3c4d5e6f
Revises: 486fff446ae3
Create Date: 2025-08-25 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from uuid import uuid4
import json

# revision identifiers, used by Alembic.
revision = '1a2b3c4d5e6f'
down_revision = '486fff446ae3'
branch_labels = None
depends_on = None

def upgrade():
    conn = op.get_bind()

    tool_id = str(uuid4())

    tool_data = {
        "id": tool_id,
        "name": "Graph Search Tool",
        "description": "This tool performs graph-based search using natural language queries. It converts user queries to Cypher statements and searches through knowledge graphs to find entities and relationships.",
        "deprecated": False,
        "tool_kind": "system",
        "tool_metadata": json.dumps({}),
        "dataset_required": True,
    }

    system_tool_data = {
        "id": tool_id,
        "tool_id": tool_id,
        "python_module": "app.tools.system_tools.logic.graph_search_logic",
        "function_name": "perform_graph_search",
        "is_async": True,
        "input_schema": "app.tools.system_tools.schemas.graph_search_schema.GraphSearchInput",
        "output_schema": "app.tools.system_tools.schemas.graph_search_schema.GraphSearchOutput",
        "function_signature": "async def perform_graph_search(input_data: GraphSearchInput) -> GraphSearchOutput"
    }

    # Insert into tools
    conn.execute(
        sa.text("""
            INSERT INTO tools (id, name, description, deprecated, tool_metadata, tool_kind, dataset_required)
            VALUES (:id, :name, :description, :deprecated, :tool_metadata, :tool_kind, :dataset_required)
        """),
        tool_data
    )

    # Insert into system_tools
    conn.execute(
        sa.text("""
            INSERT INTO system_tools (id, tool_id, python_module, function_name, is_async, input_schema, output_schema, function_signature)
            VALUES (:id, :tool_id, :python_module, :function_name, :is_async, :input_schema, :output_schema, :function_signature)
        """),
        system_tool_data
    )


def downgrade():
    conn = op.get_bind()

    conn.execute(
        sa.text("""
            DELETE FROM system_tools WHERE tool_id IN (
                SELECT id FROM tools WHERE name = :tool_name
            )
        """),
        {"tool_name": "Graph Search Tool"}
    )

    conn.execute(
        sa.text("""
            DELETE FROM tools WHERE name = :tool_name
        """),
        {"tool_name": "Graph Search Tool"}
    )
