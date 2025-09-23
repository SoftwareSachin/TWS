---
sidebar_position: 7
title: Connecting Destinations
---

# How to Connect a Destination in Amplifi

## Step 1: Access the Destinations Tab
1. Log in to Amplifi.
2. Click on the **"Destination"** tab in the top navigation bar.
3. You’ll see a list of existing destinations. You can either:
   - Click on an existing destination to view its details, or  
   - Click the **"+ Create Destination"** button in the top right corner to add a new one.  

---

## Step 2: Create a New Destination
1. Click **"+ Create Destination"**.  
2. Fill in the required fields in the form:  
   - **Name**: Enter a name for your destination.  
   - **Description**: (Optional) Add a description for better organization.  
   - **Is Active**: Toggle this to activate or deactivate the destination.  

---

## Step 3: Select Vector Type
1. In the **Select Vector** dropdown:  
   - Choose either **PG Vector** or **Databricks Vector Search**.  
2. Based on your selection, additional fields will appear:  

### If you select **PG Vector**:
In addition to filling out the required fields, you need to activate the following PostgreSQL extensions in Azure:
- `fuzzystrmatch`
- `pg_trgm`
- `uuid-ossp`
- `vector`

#### Steps to Activate Extensions in Azure Portal:
1. Go to your Azure PostgreSQL server in the Azure portal.
2. In the left sidebar, click on **"Server parameters"**.
3. In the search bar, search for `azure.extensions`.
4. Click on the `azure.extensions` parameter, and in the dropdown, select or add the following extensions:
   - `fuzzystrmatch`
   - `pg_trgm`
   - `uuid-ossp`
   - `vector`
5. Click **Save** to apply the changes.
6. After activating the extensions, proceed with entering the required fields:
   - **Host**: Enter the host URL or IP address.  
   - **Port**: Provide the port number.  
   - **Database Name**: Enter the database name.  
   - **Username**: Provide the username.  
   - **Password**: Enter the password.  

### If you select **Databricks Vector Search**:
For Databricks Vector Search, fill in the following parameters:  
- **Workspace URL**: Enter the Databricks workspace URL.  
- **Token**: Provide your Databricks access token.
- **Warehouse ID**: Enter the Warehouse ID.  
- **Database Name**: Enter the database name.  
- **Table Name**: Enter the table name.  

---

## Step 4: Test and Create the Destination
1. **Test Connection**: Click the **Test Connection** button to ensure all details are correct and the connection works.  
2. **Create**: If the connection is successful, click the **Create Destination** button.  

---

## Step 5: View Your Destination
1. Once created, you’ll be redirected to the **Destination** tab.  
2. The newly created destination should now be visible in the list.  

---

By following these steps, you can successfully connect Amplifi to your desired destination, ensuring PG Vector is correctly set up with the necessary extensions in Azure PostgreSQL.
