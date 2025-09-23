---
sidebar_position: 6
title: MCP Tool
---

# MCP Tool

The **MCP Tool** allows Amplifi agents to connect with external systems, APIs, or enterprise data sources via a standardized interface. Amplifi comes with a built-in MCP — **Tavily Web Search** — already configured and ready to use.

You can also add **external MCP tools** for your own data pipelines, enterprise search APIs, or logic.

---

## Purpose

The MCP Tool allows you to bring your **proprietary data or enterprise logic** into agent responses. It acts as a bridge between Amplifi and your custom backend — whether that's a document search API, a ticketing system, or a specialized LLM orchestrator.

This is especially useful when your data is hosted externally or governed by strict access policies, but you still want to expose it securely through a conversational interface.

---

## How It Works

When an agent is equipped with the MCP tool, it can send a query to an **external HTTP endpoint** along with the conversation context or user input. The external system (your MCP backend) processes the query, performs its logic (e.g., document search, vector search, summarization), and returns a response in a structured format.

Amplifi then renders this response in the conversation — as plain text, formatted cards, charts, or any custom UI defined on your side.

You can configure this via:

- A secure URL to your backend
- Optional headers for authentication
- A flexible response schema expected by Amplifi

---

## Typical Use Cases

- **Internal document search**: Query your proprietary vector store and serve results through chat.
- **LLM routing**: Use Amplifi as a frontend, and route complex prompts to a backend LLM orchestrator.
- **Custom AI tools**: Connect to systems like LangChain agents, knowledge graphs, or structured APIs.
- **Enterprise dashboards**: Let agents pull and display live business metrics from internal services.

---


## Best Practices

- Keep your response payload clean and well-structured.
- Include context-aware logic in your backend (e.g., summarization, filtering).
- Use versioned endpoints to support iterative improvements without breaking agents.

---

The MCP Tool makes Amplifi highly extensible — bringing your own intelligence, data, and APIs into the loop while keeping the UX conversational and seamless.
