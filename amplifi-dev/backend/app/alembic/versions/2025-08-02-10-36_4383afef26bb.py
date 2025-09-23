"""Add file explorer system tool

Revision ID: 4383afef26bb
Revises: e0e3bd08cc57
Create Date: 2025-07-17 10:36:05.523766

"""
import json
from uuid import uuid4
from alembic import op
import sqlalchemy as sa
import sqlalchemy_utils
import sqlmodel # added
from app.be_core.config import settings


# revision identifiers, used by Alembic.
revision = '4383afef26bb'
down_revision = 'b8f9c4d12e45'
branch_labels = None
depends_on = None



def upgrade():
    conn = op.get_bind()

    tool_data = {
        "id": str(uuid4()),
        "name": "File System Navigator",
        "description": """File system management tool for browsing, organizing, and inspecting files. Provides file listings, directory navigation, file metadata (size, type, path), filename searches, and basic file descriptions. Ideal for discovering what files exist, checking file properties, and understanding file organization structure. It can not find the contents in the files, it can only get you a basic description of the file (which may not contain all the information present in the file). It can not read the contents of the files, it can only get you a basic description of the file (which may not contain all the information present in the file).""",
        "deprecated": False,
        "dataset_required": True, 
        "tool_kind": "system",
        "tool_metadata": {
            "version": "1.0",
            "capabilities": [
                "list_dataset_files",
                "search_files_by_pattern",
                "get_file_metadata",
                "get_ai_generated_descriptions",
                "find_datasets_containing_file",
                "resolve_dataset_identifiers",
                "format_human_readable_sizes",
                "database_file_lookup"
            ],
            "supported_operations": ["list", "search", "get_metadata", "get_description", "find_datasets_for_file"],
        },
        "system_tool": {
            "python_module": "app.tools.system_tools.logic.file_explorer_logic",
            "function_name": "perform_file_operation",
            "is_async": True,
            "input_schema": "app.tools.system_tools.schemas.file_explorer_schema.FileExplorerInput",
            "output_schema": "app.tools.system_tools.schemas.file_explorer_schema.FileExplorerOutput",
            "function_signature": "async def perform_file_operation(input_data: FileExplorerInput) -> FileExplorerOutput",
        },
    }

    # Insert the tool
    conn.execute(
        sa.text("""
            INSERT INTO tools (id, name, description, deprecated, tool_metadata, tool_kind, dataset_required)
            VALUES (:id, :name, :description, :deprecated, :tool_metadata, :tool_kind, :dataset_required)
        """),
        {
            "id": tool_data["id"],
            "name": tool_data["name"],
            "description": tool_data["description"],
            "deprecated": tool_data["deprecated"],
            "tool_metadata": json.dumps(tool_data["tool_metadata"]),
            "tool_kind": tool_data["tool_kind"],
            "dataset_required": tool_data["dataset_required"],
        }
    )

    # Insert the system tool details
    sys = tool_data["system_tool"]
    conn.execute(
        sa.text("""
            INSERT INTO system_tools (id, tool_id, python_module, function_name, is_async, input_schema, output_schema, function_signature)
            VALUES (:id, :tool_id, :python_module, :function_name, :is_async, :input_schema, :output_schema, :function_signature)
        """),
        {
            "id": tool_data["id"],
            "tool_id": tool_data["id"],
            "python_module": sys["python_module"],
            "function_name": sys["function_name"],
            "is_async": sys["is_async"],
            "input_schema": sys["input_schema"],
            "output_schema": sys["output_schema"],
            "function_signature": sys["function_signature"],
        }
    )


def downgrade():
    conn = op.get_bind()
    tool_name = "File System Navigator"
    
    # Delete from system_tools table
    conn.execute(
        sa.text("DELETE FROM system_tools WHERE tool_id IN (SELECT id FROM tools WHERE name = :name)"),
        {"name": tool_name}
    )
    
    # Delete from tools table
    conn.execute(
        sa.text("DELETE FROM tools WHERE name = :name"),
        {"name": tool_name}
    )



