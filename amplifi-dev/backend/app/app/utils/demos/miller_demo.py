import logfire
from pydantic_ai import Agent

from app.api.deps import pydantic_ai_model_o3, tavily_client
from app.api.v1.endpoints.search import search_r2r
from app.be_core.config import settings
from app.be_core.logger import logger
from app.schemas.rag_generation_schema import IWorkspaceGenerationResponse
from app.schemas.search_schema import PerformSearchBase, VectorSearchSettings
from app.utils.llm_fns.lang_detect import construct_ssml_english

logfire.configure(token=settings.LOGFIRE_API_KEY)

agent = Agent(
    pydantic_ai_model_o3,
    # system_prompt="""
    # You are an agent with the ability to search the internet and get up-to-date info about questions.
    # Always either search the internet or scrape any provided websites first before returning an answer.
    # """,
    instrument=bool(settings.LOGFIRE_API_KEY),
)


@agent.tool_plain
def search_internet(query: str) -> dict:
    """Searches the internet for the given query."""
    return tavily_client.search(query)


@agent.tool_plain
def scrape_urls(urls: list[str]) -> dict:
    """Extracts the text of the context at the urls."""
    try:
        return tavily_client.extract(urls)
    except Exception as e:
        return {"error": str(e)}


@agent.tool_plain
async def search_rag_canada_tariffs(query: str, numb_contexts: int = 3) -> str:
    """Uses RAG to search our vector database containing up to date info on canadian tariffs, and return the top results"""
    # workspace_id = "01967d70-c416-73e1-99fc-3daa9da22fe1"
    org_id = "0195864b-e60b-7394-ae57-4dcf2f87cd65"
    dataset_ids = ["01968899-a69b-7cc1-8c01-4c51bff6ff7b"]
    _, agg_result, _ = await search_r2r(
        organization_id=org_id,
        search_settings=PerformSearchBase(
            query=query,
            dataset_ids=dataset_ids,
            vector_search_settings=VectorSearchSettings(search_limit=numb_contexts),
        ),
        use_aggregate=True,
    )
    _, agg_result_split, _ = await search_r2r(
        organization_id=org_id,
        search_settings=PerformSearchBase(
            query=query,
            dataset_ids=["0196b134-693f-7f9e-a842-53662b697949"],
            vector_search_settings=VectorSearchSettings(search_limit=numb_contexts),
        ),
        use_aggregate=True,
    )
    if agg_result is None:
        logger.warning("No results found in the database during tarrifs search.")
        return "No results found in the database."
    data_found = "\n".join(
        [
            result.text
            for result in (
                agg_result.vector_search_results
                + agg_result_split.vector_search_results
            )
        ]
    )
    return f"Found the following data in the database:\n {data_found}"


async def get_miller_agent_result(query: str) -> IWorkspaceGenerationResponse:
    result = await agent.run(query)
    return IWorkspaceGenerationResponse(
        answer=result.output,
        full_response="",
        ssml=construct_ssml_english(result.output),
    )


if __name__ == "__main__":
    # Example usage
    prompt = "What are the tariffs on importing goods from China to Canada?"
    result = agent.run_sync(
        prompt,
    )
    print(result.output)
