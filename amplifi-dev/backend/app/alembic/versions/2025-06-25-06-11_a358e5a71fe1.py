"""Add predefined system and MCP tools

Revision ID: a358e5a71fe1
Revises: 90f175c9fce9
Create Date: 2025-06-24 06:11:49.239240

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from uuid import uuid4
import json

# revision identifiers, used by Alembic.
revision = 'a358e5a71fe1'
down_revision = '90f175c9fce9'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()

    tools_data = [
        {
            "id": str(uuid4()),
            "name": "Web Search Tool",
            "description": "Retrieves up-to-date information from the internet using Brave Search — ideal for answering questions requiring current facts.",
            "deprecated": False,
            "tool_kind": "system",
            "tool_metadata": {
                "version": "1.0",
                "api_provider": "Brave Search",
                "max_results": 5,
            },
            "system_tool": {
                "python_module": "app.tools.system_tools.logic.websearch_logic",
                "function_name": "perform_web_search",
                "is_async": True,
                "input_schema": "app.tools.system_tools.schemas.websearch_schema.WebSearchInput",
                "output_schema": "app.tools.system_tools.schemas.websearch_schema.WebSearchOutput",
                "function_signature": "async def perform_web_search(input: WebSearchInput) -> WebSearchOutput",
            },
        },
        {
            "id": str(uuid4()),
            "name": "Vector Search Tool",
            "description": "Performs semantic similarity search across unstructured content such as PDFs, documents, transcripts, notes, audio, and images using vector embeddings. Use this to retrieve contextually relevant or fuzzy matches when structured querying is not appropriate.",
            "deprecated": False,
            "tool_kind": "system",
            "tool_metadata": {
                "version": "1.0",
                "search_type": "vector_similarity",
                "default_top_k": 5,
            },
            "system_tool": {
                "python_module": "app.tools.system_tools.logic.vector_search_logic",
                "function_name": "perform_vector_search",
                "is_async": True,
                "input_schema": "app.tools.system_tools.schemas.vector_search_schema.VectorSearchInput",
                "output_schema": "app.tools.system_tools.schemas.vector_search_schema.VectorSearchOutput",
                "function_signature": "async def perform_vector_search(input: VectorSearchInput) -> VectorSearchOutput",
            },
        },
        {
            "id": str(uuid4()),
            "name": "Text to SQL Tool",
            "description": "Converts natural language questions into SQL queries and executes them on structured tabular databases. Use this for numeric summaries, filtered records, or relational joins based on defined schema.",
            "deprecated": False,
            "tool_kind": "system",
            "tool_metadata": {
                "version": "1.0",
                "query_type": "natural_language_to_sql",
                "supports_visualization": True,
                "database_support": ["PostgreSQL"],
            },
            "system_tool": {
                "python_module": "app.tools.system_tools.logic.texttosql_logic",
                "function_name": "process_sql_chat_app",
                "is_async": True,
                "input_schema": "app.tools.system_tools.schemas.texttosql_schema.AgentDeps",
                "output_schema": "app.tools.system_tools.schemas.texttosql_schema.SQLProcessResponse",
                "function_signature": "async def process_sql_chat_app(request_data: AgentDeps) -> SQLProcessResponse:",
            },
        },
        {
            "id": str(uuid4()),
            "name": "Tavily MCP Tool",
            "description": "Internal tool that uses Tavily API for advanced web search and content extraction — best used for deep web intelligence tasks requiring API access.",
            "deprecated": False,
            "tool_kind": "mcp",
            "tool_metadata": {
                "version": "1.0",
                "tool_type": "internal_mcp",
                "api_provider": "Tavily",
                "supported_operations": ["web_search", "extract_content"],
                "requires_api_key": True,
            },
            "mcp_tool": {
                "mcp_subtype": "internal",
                "mcp_server_config": {
                    "tavily": {
                        "command": "python",
                        "args": ["/code/app/tools/internal_mcp/logic/tavily_mcp_server.py"],
                        "env": {
                            "TAVILY_API_KEY": "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"
                        },
                    }
                },
                "timeout_secs": 45,
            },
        },
    ]

    for tool in tools_data:
        conn.execute(
            sa.text("""
                INSERT INTO tools (id, name, description, deprecated, tool_metadata, tool_kind)
                VALUES (:id, :name, :description, :deprecated, :tool_metadata, :tool_kind)
            """),
            {
                "id": tool["id"],
                "name": tool["name"],
                "description": tool["description"],
                "deprecated": tool["deprecated"],
                "tool_metadata": json.dumps(tool["tool_metadata"]),
                "tool_kind": tool["tool_kind"],
            }
        )

        if tool["tool_kind"] == "system":
            sys = tool["system_tool"]
            conn.execute(
                sa.text("""
                    INSERT INTO system_tools (id, tool_id, python_module, function_name, is_async, input_schema, output_schema, function_signature)
                    VALUES (:id, :tool_id, :python_module, :function_name, :is_async, :input_schema, :output_schema, :function_signature)
                """),
                {
                    "id": tool["id"],
                    "tool_id": tool["id"],
                    "python_module": sys["python_module"],
                    "function_name": sys["function_name"],
                    "is_async": sys["is_async"],
                    "input_schema": sys["input_schema"],
                    "output_schema": sys["output_schema"],
                    "function_signature": sys["function_signature"],
                }
            )
        elif tool["tool_kind"] == "mcp":
            mcp = tool["mcp_tool"]
            conn.execute(
                sa.text("""
                    INSERT INTO mcp_tools (id, tool_id, mcp_subtype, mcp_server_config, timeout_secs)
                    VALUES (:id, :tool_id, :mcp_subtype, :mcp_server_config, :timeout_secs)
                """),
                {
                    "id": tool["id"],
                    "tool_id": tool["id"],
                    "mcp_subtype": mcp["mcp_subtype"],
                    "mcp_server_config": json.dumps(mcp["mcp_server_config"]),
                    "timeout_secs": mcp["timeout_secs"],
                }
            )


def downgrade():
    conn = op.get_bind()
    tool_names = [
        "Web Search Tool",
        "Vector Search Tool",
        "Text to SQL Tool",
        "Tavily MCP Tool",
    ]
    conn.execute(
        sa.text("DELETE FROM mcp_tools WHERE tool_id IN (SELECT id FROM tools WHERE name = ANY(:names))"),
        {"names": tool_names}
    )
    conn.execute(
        sa.text("DELETE FROM system_tools WHERE tool_id IN (SELECT id FROM tools WHERE name = ANY(:names))"),
        {"names": tool_names}
    )
    conn.execute(
        sa.text("DELETE FROM tools WHERE name = ANY(:names)"),
        {"names": tool_names}
    )
