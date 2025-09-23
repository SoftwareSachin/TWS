---
sidebar_position: 2
title: Connecting Data Sources
---

# Connecting Data Sources to Amplifi

Connect your cloud storage or databases to Amplifi to make your data available for chat, search, and analysis.

---

## When to Add a Data Source

You can add a data source:

- **During workspace creation** — connect while setting up  
- **After workspace creation** — go to the **Files** tab and click **+ New Files → Add Data Source**

---

## Supported Data Source Types

Amplifi currently supports:

- **Cloud Storage**: Azure Blob, AWS S3  
- **Databases**: PostgreSQL, MySQL  
- **Direct Uploads**: Drag and drop files like PDFs, images, or audio  

**Allowed File Extensions**: `docx`, `pptx`, `pdf`, `xlsx`, `csv`, `md`, `png`, `jpg`, `jpeg`, `wav`, `mp3`, `aac`

---

## How to Connect

### 1. Choose a Source

From the data source screen, select one:

- Azure Blob  
- AWS S3  
- PostgreSQL  
- MySQL  

Click **Next**.

---

### 2. Fill in Connection Details

#### **Azure Blob**
1. **Container Name**: Name of your Azure Blob container  
2. **SAS URL**: Provide the Shared Access Signature (SAS) URL to grant Amplifi access  
3. **Test Connection**:  
   - If valid, continue  
   - If not, double-check container/SAS details  

#### **AWS S3**
1. **Label**: A friendly name for your connection  
2. **Region**: Choose the correct AWS region  
3. **Access Key** and **Secret Key**: Your S3 credentials  
4. **Test Connection** before proceeding

#### **PostgreSQL**
1. **Database Name**  
2. **Host Name** and **Port** (usually 5432)  
3. **Username** and **Password**  
4. Click **Next** to test and validate connection

#### **MySQL**
1. **Database Name**  
2. **Host Name** and **Port** (usually 3306)  
3. **Username** and **Password**  
4. *(Optional)* Enable **SSL Mode**  
5. Click **Next** to test and validate connection

---

## After Connecting

- For **file-based sources**: Files are auto-imported to your workspace  
- For **databases**: Amplifi enables querying structured data directly  

---

## Tips & Troubleshooting

- Always click **Test Connection** to verify setup before proceeding  
- Recheck hostnames, ports, credentials, or SAS URLs if there's a failure  
- Ensure databases are reachable from Amplifi’s IP (if firewalled)

---

Once connected, your data becomes searchable, secure, and ready for AI-powered interaction in Amplifi.
