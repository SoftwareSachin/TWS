export const agentic_constants = {
  DATASET_TYPE: {
    UNSTRUCTURED: "unstructured",
    STRUCTURED: "structured",
    SQL: "sql",
  },

  TRUTH_VALUES: {
    TRUE: true,
    FALSE: false,
  },
  TOOL_TYPE: {
    LABEL: {
      MCP: "MCP",
      SYSTEM_TOOL: "System Tool",
    },
    VALUE: {
      MCP: "MCP Server",
      SYSTEM_TOOL: "System Tool",
    },
    MCP_TOOL: "MCP",
    SYSTEM_TOOL: "System Tool",
  },
  TOOL_KIND: {
    MCP: "mcp",
    SYSTEM: "system",
  },
  MCP_SUBTYPE: {
    INTERNAL: "internal",
    EXTERNAL: "external",
  },

  EDIT: {
    AGENT: "Edit Agent",
    TOOL: "Edit Tool to Workspace",
  },
  CREATE: {
    AGENT: "Create New Agent",
    TOOL: "Add Tool to Workspace",
  },

  TOOL_NAMES: {
    TEXT_TO_SQL: "Text to SQL Tool",
    VECTOR_SEARCH: "Vector Search Tool",
    FILE_SYSTEM: "File System Navigator",
  },
};
