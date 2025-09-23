---
sidebar_position: 10
title: Configure MCP
---

# Configure MCP

Amplifi allows you to integrate **Model Context Protocol (MCP)** tools to extend its core functionality.

> 🧠 *MCPs let you run custom logic for parsing, enrichment, or external task execution inside your Amplifi workflows.*

When you navigate to the **MCP** tab in the left navigation, you’ll see a built-in internal MCP tool called **Tavily MCP Tool**. This tool is already pre-configured by Amplifi and ready for immediate use.

In addition to this, you can add your own **external MCPs** for custom use cases.

---

## Adding an External MCP

To add a new external MCP:

1. Go to the **MCP** tab from the left navigation bar.
2. Click **Add External MCP** on the top right.
3. You’ll see a configuration form like this:


4. Fill in the fields:
   - **MCP Name** – A short, unique identifier.
   - **MCP Description** – What this MCP does.
   - **Paste Config Code** – A JSON configuration block (see below).

---

## Configuration Format

To configure your external MCP, please use the following JSON format:

```json
{
  "your_mcp_name": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/your-mcp-package", "arg1"],
    "env": {}
  }
}

### Notes

- Replace `"your_mcp_name"` with the identifier you wish to assign (e.g., `"custom_mcp"`).
- Replace `"@modelcontextprotocol/your-mcp-package"` with your actual MCP package.
- Modify `"arg1"` to the appropriate command required by your MCP.
- Use the `"env"` section to define any environment variables your MCP requires.

> ⚠️ Only MCPs that can be executed with the `npx` command are currently supported.
