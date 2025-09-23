---
sidebar_position: 1
title: Create Datasets
---

# What Is a Dataset?

A dataset is a collection of related files that you can search, chat with, and analyze in Amplifi. It gives structure to your data and is essential for organizing large amounts of information.

## Creating Your First Dataset

Datasets in Amplifi help you group files into collections, making data easier to search, organize, and analyze. Follow this guide to create your first dataset.

## Creating a Dataset

### Step 1: Access the Dataset Creation Screen

1. Open your workspace and go to the **Datasets** tab.  
2. Click the **+ New Dataset** button in the top-right corner.

### Step 2: Configure Dataset Details

1. Enter a **Name** and **Description** for your dataset.

2. Choose your **File Option**:  
   - **Select File**: Pick specific files for your dataset.  
     - Click the dropdown to open the file selector.  
     - Check the files you want to add, then click **Add files**.  
   - **Ingest All New Files From Source**: Automatically ingest all new files from the specified source when uploading data to the workspace.  

### Step 3: Configure Chunking (Optional)

After creating your dataset, you can configure chunking to break down large documents into smaller, searchable pieces.

### Step 4: Start Ingestion

Ingestion processes your files, making them ready for search and chat.

Once ingestion is complete, click on the dataset to view the chunks of all files in that dataset and their vector embeddings.  

[Learn how to start ingestion →](../how-to-guides/configuring-ingestion)

### 💡 Knowledge Graph Creation

After ingestion, Amplifi allows you to build a Knowledge Graph from your dataset to extract and visualize entities and their relationships.

You will see a button labeled **+ Add Graph to Dataset**. Clicking it begins the graph generation process, where entities and relationships are extracted automatically.

👉 [Learn how to create a graph →](../how-to-guides/creating-graph)

## Managing Datasets

- **Edit Dataset**: You can update the dataset to add or remove files, change its name or description, and modify chunking settings.
- **Re-Ingest Files**: To apply a different chunking configuration, use the edit functionality. Modify the chunking settings and re-ingest the files as needed.
- **Delete Dataset**: Remove datasets no longer needed to keep your workspace organized.

## Tips for Organizing Datasets

- Group files by project, topic, or team to make navigation easier.  
- Use meaningful names to help team members quickly understand dataset content.  
- Regularly review datasets to keep data relevant and organized.

You’ve now created your first dataset! 🚀 Next, dive into configuring chunking and ingestion to make the most of your data.
