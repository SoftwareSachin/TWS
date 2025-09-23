import json

from app.api.deps import get_gpt41_client
from app.api.v1.endpoints.search import search_r2r
from app.be_core.logger import logger
from app.models.chat_history_model import ChatHistory
from app.schemas.rag_generation_schema import IWorkspaceGenerationResponse, RagContext
from app.schemas.search_schema import PerformSearchBase, VectorSearchSettings
from app.utils.llm_fns.lang_detect import construct_ssml_english

tools = [
    {
        "name": "search_company_data",
        "type": "function",
        "function": {
            "name": "search_company_data",
            "description": "Search for the latest data about a company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "enum": [
                            "Natural Resources Canada",
                            "Cenovus Energy",
                            "BP",
                            "ExxonMobil",
                            "MEG Energy",
                            "Ovintiv",
                            "TAQA",
                            "Suncor Energy",
                            "TC Energy",
                            "LNG Canada",
                            "Kingston Midstream",
                            "Tundra Oil & Gas",
                            "Whitecap Resources",
                            "Tidewater",
                            "Strathcona Resources",
                            "PETRONAS",
                            "Pembina",
                            "Imperial",
                            "Trans Mountain",
                            "Plains Midstream",
                            "KUFPEC",
                            "Torxen",
                            "Prospera Energy",
                            "Canlin Energy",
                            "NOVA Chemicals",
                            "North West Refining",
                            "Ember",
                            "Parkland",
                            "Shell",
                            "Enbridge",
                            "Keyera",
                            "NorthRiver Midstream",
                            "CNOOC",
                            "TransAlta",
                            "Obsidian Energy",
                        ],
                        "description": "Select one of the supported company names.",
                    },
                    "query": {
                        "type": "string",
                        "description": "The query or question you want to ask about the company, try to make it as specific as possible.",
                    },
                },
                "required": ["company_name", "query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

company_map = {
    "Natural Resources Canada": "xxx",
    "Cenovus Energy": "019640b9-b19a-79af-9a19-0efd097270c6",
    "BP": "01963bbd-b060-79b8-84e5-f74f3a817752",
    "ExxonMobil": "019644cf-7f0d-71fe-aa27-7a5095daa821",
    "MEG Energy": "019644da-71a7-722b-9283-b8d3cb5415c7",
    "Ovintiv": "019644e2-922c-737b-800d-1a2e652a8da2",
    "TAQA": "xxx",
    "Suncor Energy": "019644fc-6414-738a-a271-f14b07bc28bd",
    "TC Energy": "019644fd-282c-7e1f-aedd-1af0dbf1c6d0",
    "LNG Canada": "019644d4-0424-7982-80c5-5c6d37d20288",
    "Kingston Midstream": "xxx",
    "Tundra Oil & Gas": "xxx",
    "Whitecap Resources": "xxx",
    "Tidewater": "xxx",
    "Strathcona Resources": "019644fb-9bea-7866-9b70-6e8430424545",
    "PETRONAS": "019644f6-1026-7725-b22e-9872d7c1a0e9",
    "Pembina": "019644f5-486a-79bc-8e76-0bc9d6e0f016",
    "Imperial": "019644d1-ea31-7ee7-ba1d-7381d38d71d1",
    "Trans Mountain": "xxx",
    "Plains Midstream": "019644f7-034a-77e3-b213-75406327c694",
    "KUFPEC": "xxx",
    "Torxen": "xxx",
    "Prospera Energy": "xxx",
    "Canlin Energy": "xxx",
    "NOVA Chemicals": "xxx",
    "North West Refining": "xxx",
    "Ember": "xxx",
    "Parkland": "019644f4-7c9c-7afd-8cc5-5603a243ce02",
    "Shell": "019644f7-b0bd-7e19-9fc9-3dfbe7640575",
    "Enbridge": "019644c8-99a4-7707-927a-e270d0a2def6",
    "Keyera": "019644d3-5252-75f9-b173-e0990089f2d6",
    "NorthRiver Midstream": "xxx",
    "CNOOC": "019640bd-bca6-75ed-9d48-a0398b134be4",
    "TransAlta": "019644fd-d254-788f-a746-3e5099e5b230",
    "Obsidian Energy": "xxx",
}

client = get_gpt41_client()


async def search_company_report(
    query: str, history: list[ChatHistory]
) -> IWorkspaceGenerationResponse:
    messages = [
        {
            "role": "system",
            "content": """You are a chatbot capable of searching various company reports.
            If you don't have data on a specific company, please respond that you don't have information on the company, and end your response there. No need to provide information if you have no info.
            Don't use special symbols such as "#" or "*" in your response, the user is not able to render markdown.""",
        }
    ]
    for line in history:
        messages.append({"role": "user", "content": line.user_query})
        messages.append({"role": "assistant", "content": line.llm_response})
    messages.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.1,
        messages=messages,
        tools=tools,
    )
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        logger.warning("No tool calls found in the response.")
        return IWorkspaceGenerationResponse(
            answer=response.choices[0].message.content,
            full_response=str(response),
            contexts_found=[],
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            ssml=construct_ssml_english(response.choices[0].message.content),
        )
    messages.append(
        {
            "role": "assistant",
            "tool_calls": tool_calls,
        }
    )
    original_input_tokens = response.usage.prompt_tokens
    original_output_tokens = response.usage.completion_tokens
    contexts_found = []
    for tool_call in tool_calls:
        arugments = json.loads(tool_call.function.arguments)
        company_name = arugments["company_name"]
        query = arugments["query"]
        logger.debug(f"Searching for {company_name} with query: {query}")
        if (
            company_map.get(company_name, None)
            and company_map.get(company_name, None) != "xxx"
        ):
            _, result, _ = await search_r2r(
                organization_id="76d1d9a1-6a82-4051-8a60-762025764995",
                search_settings=PerformSearchBase(
                    query=query,
                    vector_search_settings=VectorSearchSettings(),
                    dataset_ids=[company_map.get(company_name, None)],
                ),
                use_aggregate=True,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "\n\n".join(
                        [vec_result.text for vec_result in result.vector_search_results]
                    ),
                }
            )
            contexts_found.extend(
                RagContext(
                    text=vec_result.text,
                    page_numb=vec_result.unstructured_page_number,
                )  # add further file metadat later
                for vec_result in result.vector_search_results
            )
        else:
            logger.warning(f"Company {company_name} not found in the map.")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "No data found for the company.",
                }
            )
    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.1,
        messages=messages,
    )
    return IWorkspaceGenerationResponse(
        answer=response.choices[0].message.content,
        full_response=str(response),
        contexts_found=contexts_found,
        input_tokens=response.usage.prompt_tokens + original_input_tokens,
        output_tokens=response.usage.completion_tokens + original_output_tokens,
        ssml=construct_ssml_english(response.choices[0].message.content),
    )
