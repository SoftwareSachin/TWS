"""
Simple Tavily MCP Server for Web Search
"""

import asyncio
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from environment
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
# Initialize Tavily client
tavily_client = None
if TAVILY_API_KEY:
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        logger.info("Tavily client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Tavily: {e}")
else:
    logger.error("TAVILY_API_KEY not found in environment variables")

# Create MCP server
server = FastMCP("Tavily Web Search")


@server.tool()
async def web_search(query: str, max_results: int = 5) -> dict:
    """
    Search the web using Tavily API

    Args:
        query: Search query
        max_results: Maximum number of results (default: 5)

    Returns:
        Search results with sources
    """
    if not tavily_client:
        return {"error": "Tavily client not initialized. Check TAVILY_API_KEY."}

    try:
        result = tavily_client.search(query=query, max_results=max_results)
        logger.info(
            f"Search completed: '{query}' - {len(result.get('results', []))} results"
        )
        return result
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"error": f"Search failed: {str(e)}"}


@server.tool()
async def extract_content(urls: list[str]) -> dict:
    """
    Extract content from URLs using Tavily

    Args:
        urls: List of URLs to extract content from

    Returns:
        Extracted content from URLs
    """
    if not tavily_client:
        return {"error": "Tavily client not initialized. Check TAVILY_API_KEY."}

    if not urls:
        return {"error": "No URLs provided"}

    try:
        result = tavily_client.extract(urls)
        logger.info(f"Content extracted from {len(urls)} URLs")
        return result
    except Exception as e:
        logger.error(f"Content extraction failed: {e}")
        return {"error": f"Content extraction failed: {str(e)}"}


def main():
    """Run the MCP server"""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    if not TAVILY_API_KEY:
        logger.error("TAVILY_API_KEY environment variable is required")
        sys.exit(1)

    logger.info("Starting Tavily MCP Server...")
    server.run()


if __name__ == "__main__":
    main()
