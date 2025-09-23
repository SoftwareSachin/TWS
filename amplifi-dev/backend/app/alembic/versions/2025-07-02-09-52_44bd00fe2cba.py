"""Add Plot_generate system tool

Revision ID: 44bd00fe2cba
Revises: 929d9dd320e5
Create Date: 2025-07-02 09:52:13.656396

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from uuid import uuid4
import json

# revision identifiers, used by Alembic.
revision = '44bd00fe2cba'
down_revision = '929d9dd320e5'
branch_labels = None
depends_on = None

def upgrade():
    conn = op.get_bind()

    tool_id = str(uuid4())

    tool_data = {
        "id": tool_id,
        "name": "Visualization Tool",
        "description": "This tool is basically used for returning plotly html and plotly code based on the provided context in the recent memory to produce the graph or visualization charts. It is used to generate visualizations from the context provided by the user.",
        "deprecated": False,
        "tool_kind": "system",
        "tool_metadata": json.dumps({}),
    }

    system_tool_data = {
        "id": tool_id,
        "tool_id": tool_id,
        "python_module": "app.tools.system_tools.logic.plot_logic",
        "function_name": "generate_plotly_from_llm",
        "is_async": True,
        "input_schema": "app.tools.system_tools.schemas.plot_schema.PlotMockInput",
        "output_schema": "app.tools.system_tools.schemas.plot_schema.PlotMockOutput",
        "function_signature": "async def generate_plotly_from_llm(request_data: PlotMockInput) -> PlotMockOutput"
    }

    # Insert into tools
    conn.execute(
        sa.text("""
            INSERT INTO tools (id, name, description, deprecated, tool_metadata, tool_kind)
            VALUES (:id, :name, :description, :deprecated, :tool_metadata, :tool_kind)
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
        {"tool_name": "Plot_generate"}
    )

    conn.execute(
        sa.text("""
            DELETE FROM tools WHERE name = :tool_name
        """),
        {"tool_name": "Plot_generate"}
    )