---
sidebar_position: 3
title: Destinations
---

# Destinations

Destinations in Amplifi are essential endpoints where vectorized data—processed during ingestion—gets transferred and stored. They enable smooth integration between Amplifi and your chosen database, ensuring vectors are accessible for further processing, querying, and AI-powered insights.

## What Are Destinations?

Destinations act as repositories for storing vector data extracted from your source content. After ingestion, the data is converted into vectors, and these vectors need to be transferred to a designated storage solution. Destinations provide a secure and streamlined way to achieve this, connecting Amplifi to user databases.

## Why Use Destinations?

Destinations ensure that your vectorized data is stored where you need it, enabling downstream processes like vector search, retrieval, and analysis. Key benefits include:

- 🚀 **Seamless Integration**: Connect Amplifi to your preferred vector database.  
- 🔐 **Secure Transfer**: Transfer vectors securely without compromising data integrity.  
- 📡 **Accessible Data**: Make vectorized data readily available for further querying and insights.  

## Available Destination Types

Amplifi supports multiple destination types to suit different infrastructure needs:

| Destination Type        | Description                                           | Best For                                |
|-------------------------|-------------------------------------------------------|----------------------------------------|
| **PG Vector**            | Integrates with PostgreSQL to store vector embeddings  | Organizations using PostgreSQL           |
| **Databricks Vector Search** | Integrates with Databricks for scalable vector search | Teams working with Databricks environments |

## How Destinations Work

1. **Connection**: Choose a destination type and provide the necessary credentials.  
2. **Validation**: Test the connection to ensure proper integration.  
3. **Storage**: Once created, vectors are transferred to the selected destination post-ingestion.  

## Setting Up Your First Destination

Setting up a destination is simple. Follow the step-by-step guide below to get started:  

[Learn how to connect your first destination →](../how-to-guides/connecting-destination)  

## Security & Privacy

Amplifi ensures security at every step with measures like:

- **Encrypted Connections**: All data transfers are encrypted.  
- **Credential Protection**: Access credentials are securely stored and never exposed.  
- **Access Controls**: Only authorized users can create, manage, and access destinations.  

## Best Practices

- Test the destination connection before creating it to avoid configuration errors.  
- Use clear and descriptive names for each destination to avoid confusion.  
- Regularly update credentials and perform access audits.  

By setting up the right destination, you ensure that your vector data is securely stored and accessible, unlocking the full potential of Amplifi’s search and analysis capabilities. 🌟
