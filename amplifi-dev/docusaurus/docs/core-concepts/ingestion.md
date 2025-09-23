---
sidebar_position: 4
title: Ingestion
---

# Understanding Ingestion

Ingestion is the process of preparing your data for search, chat, and analysis in Amplifi. It breaks large files into smaller chunks, generates vector embeddings, and indexes the data — making it faster and more efficient to retrieve relevant information.

## Why Ingestion Matters

Ingesting data is essential to ensure that large, unstructured documents become easily searchable and accessible for AI-driven queries. Key benefits include:

- **Faster Search Results**: Smaller chunks enable quicker lookups.  
- **Improved Accuracy**: Overlapping chunks ensure context is preserved across boundaries.  
- **Scalability**: Handles large datasets by processing them into manageable units.  
- **Semantic Understanding**: Vector embeddings allow the system to understand meaning, not just keywords.  

---

## The Ingestion Process in Amplifi

When you upload files into a dataset, Amplifi performs several steps to prepare the data:

1. **File Splitting**: Breaks large files into smaller segments to make them easier to process.  
2. **Chunk Processing**: Each chunk is converted into vector embeddings for semantic understanding.  
3. **Indexing**: Chunks and embeddings are stored in a searchable index.  
4. **Ready for Querying**: The system retrieves relevant chunks when you perform a search or ask a question.  

---

## Chunking: Breaking Data into Meaningful Units

Chunking is the process of dividing large files into smaller, meaningful segments. This helps manage data size, maintain context, and improve retrieval accuracy.


### Chunk Size and Overlap

- **Chunk Size**: Determines the size of each chunk. Larger chunks retain more context, but smaller chunks improve search latency. As a rule of thumb a page contains roughly 1000 tokens. So with the default setting of 2500 tokens, this covers 2.5 pages of content in a pdf. For most reports (user manuals, 10k filings, market research reports, invoices etc.) this chunk size would maintain enough context for an LLM to retrieve the chunk and reason on the chunk to provide an accurate answer to the input query.  
- **Chunk Overlap**: Ensures chunks share some content with their neighbors, preserving context across boundaries. By default a 10% chunk overlap on chunk size is recommended as a starting point. If the document contains complex concepts that span pages, then increasing the chunk overlap to upto 50% would make sense.

## Vector Embeddings: Unlocking Semantic Search

Vector embeddings transform text into numerical representations, enabling semantic search by capturing meaning rather than just keywords. Amplifi uses:

- **OpenAI Embeddings**: High-accuracy embeddings that understand context and meaning.

These embeddings empower Amplifi to:

- Retrieve the most relevant chunks.
- Improve search accuracy by understanding context.
- Enable natural language queries for deeper insights.