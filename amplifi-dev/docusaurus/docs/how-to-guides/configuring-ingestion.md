---
sidebar_position: 4
title: Configure Ingestion
---

# Ingesting Datasets

Ingesting datasets is the process of making your files searchable and ready for analysis in Amplifi. Follow this guide to configure ingestion settings and start the process.

## Step 1: Configure File Ingestion

After creating your dataset, the **Configure File Ingestion** screen will appear.

1. **Enter Configuration Name**:  
   - This name helps identify your custom configuration for ingestion.  
   - Example: *Customer Support Ingestion*.

2. **Set Chunk Size and Overlap**:  
   - **Chunk Size**: Defines the size of each chunk (default is *2500*).  
   - **Chunk Overlap**: Controls overlap between chunks to retain context (default is *250*).

<!--
3. **Select Embedding Model**:  
   Choose the embedding model used to process the dataset:  
   - Example: *OpenAI Text Embedding 3 Large* (recommended for faster processing).
-->

## Step 2: Start Ingestion

Once the configuration is set:

1. Click the **Start Ingestion** button to begin processing the files.  
2. A loading screen will appear, indicating that ingestion has started. The process runs in the background, allowing uninterrupted access to other features while it completes.

## Managing Ingestion

- **Monitor Progress**: Amplifi shows the ingestion status, so you can track progress.  
- **Re-Ingest Files**: If ingestion fails, you will have the option to re-ingest the affected files.  
- **Edit and Re-Ingest**: You can reconfigure the ingestion settings by editing the dataset and starting the ingestion again.  
- **View Chunks and Vectors**: Once ingestion is complete, click on the dataset name to view the chunks and vector embeddings of all files in that dataset.  
- **Add Graph to Dataset**: Upon completion of the ingestion process, a **+ Add Graph to Dataset** button will become available. Selecting this option will initiate the creation of a knowledge graph. A processing screen will be displayed to indicate the ongoing process, which runs in the background, allowing you to continue utilizing other features within Amplifi uninterrupted.

Ingesting datasets ensures your files are indexed and ready for search, chat, and insights. 🚀
