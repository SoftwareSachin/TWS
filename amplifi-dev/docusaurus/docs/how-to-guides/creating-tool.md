---
sidebar_position: 9
title: Creating Tools
---

# Creating Tools

Amplifi allows you to create and integrate intelligent tools into your workspace. These tools can perform tasks such as answering questions using vector or SQL search, running web queries, or visualizing data.

## Accessing the Tool Section

1. Go to your **Workspace**.  
2. Click the **Tools** tab from the sidebar.  
3. You will see all existing tools listed here.  
4. Click **Add Tool to Workspace** to begin creating a new tool.

## Creating a New Tool

Once you click **Add Tool to Workspace**, you'll see the **Tool Creation** form.

### 1. Tool Name  
- Give your tool a name.  

### 2. Tool Description  
- Describe what your tool is meant to do.  

### 3. Associate Tool With  
You must choose whether the tool is a:

- **System Tool**: Pre-integrated tools provided by Amplifi (e.g., Text to SQL, Vector Search, etc.)
- **MCP Server Tool**: External tool connected from your custom MCP setup.

--- 

## System Tool Configuration

If you select **System Tool**, choose from the following available types:

> 📘 [Learn more about System Tools](../core-concepts/system-tools)

### A. Web Search Tool  
- Uses external search engines to answer user queries in real time.

### B. Vector Search Tool  
- Used to perform semantic search over unstructured data (e.g., PDFs, documents).  
- **You must select one or more unstructured datasets** where the search should be performed.

### C. Text to SQL Tool  
- Converts user questions into SQL queries for structured datasets.  
- **You must add one or more structured datasets** that you want to query using this tool.

### D. Visualization Tool  
- Allows querying and visualizing datasets through charts and dashboards.  
- **You can add datasets of any type** that you want to visualize using this tool.

After selecting the system tool type and linking required datasets, click **Submit**.

---

## MCP Tool Configuration

If you select **MCP Server**, choose a configured MCP tool:

1. Select the MCP server from the dropdown.
2. Choose the available tool configured under that MCP.
3. For more information on connecting MCP tools, see:  
   👉 [Learn how to add your MCP tool](../how-to-guides/configuring-mcp)

Once done, click **Submit** to create the tool.

---

## Using Your Tool

- The new tool will now appear in your **Tools** list.
- You can:
  - View or edit the tool’s details.
  - Associate it with agents.
  - Delete it if it's no longer needed.

Tools make it easy to extend Amplifi’s capabilities with customized behavior tailored to your use cases. 🚀
