---
sidebar_position: 7
title: System Tools
---

# System Tools

System tools are built-in capabilities that empower Amplifi agents to perform specialized tasks — such as searching the internet, querying structured data, or generating charts — beyond just language generation.

These tools act as modular plugins that agents can call to retrieve or process information intelligently. They are configurable, reusable, and built to solve domain-specific tasks with precision.

---

## Why Use System Tools?

System tools enhance agent capabilities by enabling them to:

- 🔍 **Search real-time information from the web**
- 📚 **Find relevant content from large document sets**
- 🧠 **Understand and query tabular data**
- 📊 **Turn insights into visual summaries**

By attaching tools to an agent, you enable it to handle queries using the right approach based on the task.

---

## Core System Tools in Amplifi

### 🟦 Web Search Tool

**Purpose**: Fetch the most up-to-date answers from the live web.  
Great for time-sensitive questions or topics not covered in your internal documents — like “What’s the current repo rate?” or “Who won the last Formula 1 race?”

---

### 🟪 Vector Search Tool

**Purpose**: Find semantically relevant content from ingested documents.  
Perfect for retrieving answers from PDFs, meeting notes, screenshots, or transcripts — even when keywords don’t exactly match.

---

### 🟨 Text to SQL Tool

**Purpose**: Understand natural language questions and generate meaningful answers from your structured databases.  
Useful for answering things like “What were last month’s total sales?” or “List the top 3 performing regions this quarter.”

---

### 🟧 Visualization Tool

**Purpose**: Convert insights into dynamic charts and graphs.  
Great for turning summaries into visual formats — like “Show a trend of sales over 6 months” or “Plot leads by region.”

---

## Note

Tools like the **MCP Tool** are *external integrations* you bring into Amplifi — they are not built-in system tools, but powerful ways to extend Amplifi with your own enterprise logic or APIs. See the [MCP Tool](../core-concepts/mcp-tool) page for details.

---

## How Agents Use Tools

When a tool is enabled for an agent, it becomes part of the agent’s reasoning system. At runtime:

1. The agent analyzes the user’s question.
2. It chooses the right tool based on context.
3. It runs the tool in the background and responds with a refined, insightful answer.

---

## Best Practices

- 🎯 Assign tools based on what the agent needs to solve — not all tools are always required.
- 🧪 Test each tool with a few trial queries to fine-tune agent instructions.
- 🧩 Combine tools with well-crafted system prompts to maximize value.

---

System tools transform agents into smart problem-solvers — capable of reasoning, searching, and responding with data-backed intelligence.
