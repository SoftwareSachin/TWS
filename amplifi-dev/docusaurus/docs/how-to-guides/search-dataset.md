# Searching in Datasets

Once datasets are ingested, Amplifi allows you to efficiently search through your files. This feature helps you extract relevant insights quickly by querying the ingested datasets.

## Accessing the Search Page

1. **Navigate to the Search Tab**  
   Go to your **Workspace** and select the **Search** tab from the sidebar.

2. **Select Datasets**  
   Use the **Dataset** dropdown to choose one or more datasets you want to search through.

3. **Enter Your Query**  
   Type your question or search prompt in the query field. 

4. **Search Method**  
   Amplifi uses **Cosine Distance** to rank results. This is a standard method for measuring the relevance between query and document embeddings by capturing the directional alignment of their meanings.

5. **Run the Search**  
   Click the **Search** button to view results ranked by relevance.

---

## Understanding the Results

Each search result includes:
- A snippet of matched content from the dataset
- A **Search Score**, indicating how closely it matches your query (higher is more relevant)

---

## Viewing Performance Metrics

At the top-right, you’ll see a checkbox: **Show metrics per dataset**

### ✅ When enabled:
You’ll see **separate metrics for each dataset**:
- **Precision**: Measures how relevant the results from that dataset are to your query.
- **NDCG** (Normalized Discounted Cumulative Gain): Reflects the quality of result ranking — higher means the top results are more relevant.
- **Latency**: Time taken to return results for the search (in seconds).

This helps you compare how well each dataset is performing.

### ⛔ When disabled:
You’ll see a **single aggregated view** showing overall:
- **Precision**
- **NDCG**
- **Latency** 
across all selected datasets.

---

## Summary

- You can search across multiple datasets using semantic similarity.
- Results are ranked using cosine-based text similarity.
- Performance metrics help assess dataset quality and search effectiveness.

Use this interface to find accurate insights, helping you quickly find relevant data for further analysis. 
