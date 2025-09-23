---
sidebar_position: 8
title: Create Workflows
---

# What Is a Workflow?

A workflow in Amplifi helps transfer vector embeddings generated from ingestion into a destination of your choice. It automates the process of moving processed data, ensuring your destination system stays updated with the latest ingested vectors.

## Creating a Workflow

Workflows in Amplifi are essential for streamlining the transfer of data from datasets to destinations. Follow this guide to create your first workflow.

### Step 1: Access the Workflow Creation Screen

1. Log in to Amplifi.
2. Switch to the **Workflows** tab at the top of the screen.
3. Click the **+ Create Workflow** button in the top-right corner.

### Step 2: Configure Workflow Details

1. Enter a **Name** and **Description** for your workflow.  
   - Example: *Customer Support Ingestion* or *Product Data Transfer*.

2. **Select Dataset**:  
   - Choose a dataset containing the vectors you want to transfer.  
   - [Learn more about creating datasets →](../how-to-guides/datasets.md)

3. **Connect Destination**:  
   - Pick a destination to send the vectors.  
   - [Learn more about connecting destinations →](../core-concepts/destinations.md)

### Step 3: Set the Schedule

You can schedule the workflow to run automatically:

1. Choose a **Trigger Time**:  
   - Options: Morning, Afternoon, Evening.  

2. Set the **Frequency**:  
   - Options: Daily, Weekly, Monthly.  

3. Click **Next** to create the workflow.

### Step 4: View and Manage Workflows

Once created, the workflow appears under the **Workflows** tab. From here, you can:

- **View Workflow**: See workflow details, the selected dataset, destination, and schedule.  
- **Edit Workflow**: Modify the dataset, destination, or schedule.  
- **Pause/Resume Workflow**: Control the workflow’s activity.  
- **Run History**: Review past runs and their statuses.

## Tips for Organizing Workflows

- Name workflows clearly to identify their purpose at a glance.  
- Group workflows by team or project for easy navigation.  
- Regularly review workflows to ensure they stay relevant and efficient.  

You’ve now created your first workflow! 🚀 Explore further by connecting destinations and managing schedules.  
