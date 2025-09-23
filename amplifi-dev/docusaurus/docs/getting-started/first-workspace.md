---
sidebar_position: 2
title: Creating Your First Workspace
---

# Creating Your First Workspace

Workspaces are the heart of Amplifi, where your team collaborates and organizes projects. This guide will walk you through creating and configuring your first workspace.

## What Is a Workspace?

Before diving in, it's helpful to understand what an Amplifi workspace offers:

- **Dedicated environment** for specific projects or departments
- **Customizable permissions** controlling who can access what
- **Project-specific data sources** and configurations
- **Collaborative features** for team members

## Creating a New Workspace

### Step 1: Access the Workspace Creation Screen

1. Log in to your Amplifi account
2. You will land on the Workspace Dashboard.
3. Click the **+ Create Workspace** button in the top-right corner


### Step 2: Configure Basic Workspace Details

1. Enter a **Workspace Name** that clearly identifies its purpose
   - Example: "Marketing Analytics" or "Customer Support Knowledge Base"
2. Add a **Description** explaining what the workspace is for
3. Click the **Submit** button to create your workspace.


### Step 3: Connect to a Data Source

After creating the workspace, you'll be redirected to the **Upload files to connect to a data source** screen.  
1. Choose how you want to add data:
   - **Upload Files**: Drag & drop files or click **Upload file**.  
     - **Allowed File Extensions**: `docx`, `pptx`, `pdf`, `xlsx`, `csv`, `md`, `png`, `jpg`, `jpeg`, `wav`, `mp3`, `aac`  
   - **Connect to Cloud Storage or Databases**:  
     - Click **Azure Blob** to connect with Microsoft Azure  
     - Click **AWS S3** to connect with Amazon S3  
     - Click **PostgreSQL** or **MySQL** to connect structured data sources  
2. After adding your data, click **Next**. 

[Learn more about data sources →](../core-concepts/data-sources) 


### Step 4: Workspace Dashboard

1. Review your workspace settings
2. Click **Create Workspace**
3. You'll be automatically directed to your new workspace

## Configuring Your Workspace

Once data is added, you’ll be redirected back to the Workspace Dashboard, where your new workspace will appear.
- Click on the workspace tile to open it.

## Navigating Your New Workspace

### Explore the Files Tab
After opening your newly created workspace, you’ll land on the Files tab. This is where you can manage and organize your uploaded data.
1. View Uploaded Files
2. Add More Files by clicking the **+ New Files** button in the top-right corner to upload additional files.

### Create a Dataset
Datasets help organize your files into collections, making them easier to search and manage.

[Learn more about datasets →](../how-to-guides/datasets) 

### Search Your Files
Amplifi makes it easy to find content within your files.

[Learn more about searching datasets →](../how-to-guides/search-dataset.md) 

### MCP

- Amplifi provides **internally configured MCP tools** ready to use out of the box.
- You can also **configure external MCP tools** to suit your workflows.

[Learn more about configuring MCP tools →](../how-to-guides/configuring-mcp)

### Tools

Amplifi includes a library of **built-in tools** for:
- Searching large knowledge bases
- Generating visualizations
- Performing live web searches

These tools can be plugged into agents or used independently.

[Learn more about creating tools →](../how-to-guides/creating-tool)


### Agents

Create and configure intelligent agents that can automate workflows or answer questions using your data.
- Customize behavior, personality, and tools available to each agent.

[Learn about agents →](../how-to-guides/creating-agent)

### Chat With Your Data
Want insights directly from your data? Amplifi lets you chat with your documents.

[Learn more about Amplifi powered chatbot →](../how-to-guides/creating-chatapp) 

### Manage Users

Amplifi provides intuitive user management to help you control workspace access.

- Click the **Users** tab in the left-hand menu to view users with access to the current workspace.  
- By default, only the workspace creator is listed and has access initially.  
- Users in this list are restricted to **that specific workspace**.  
- **Organization admins**, however, automatically have access to all workspaces.  
- To add more users to a workspace, click the **+ Add User** button in the top-right corner.  
- Select users from the organization-wide list to grant them access.  
- This list only includes users already added to the organization.

[Learn how to invite users to your organization →](../how-to-guides/add-new-users)  

> **Note:** Admins do not need to be added manually to each workspace—they already have full access.


## Best Practices for Workspace Organization

For maximum effectiveness:

- **Use clear naming conventions** for workspaces and data sources
- **Start small** with one focused use case before expanding
- **Document your workspace purpose** in the description
- **Regularly review access permissions** to maintain security
- **Create separate workspaces** for distinct business functions

## Next Steps

Now that you've created your first workspace:

- [Connect your first data source](../how-to-guides/connecting-data-sources)
- [Configure your first destination](../how-to-guides/connecting-destination)

<!-- - [Learn about navigating the Amplifi interface](./navigation-basics)
- [Understand AI capabilities](../core-concepts/ai-capabilities) -->
