from vanna.openai import OpenAI_Chat
from vanna.pgvector import PG_VectorStore

from app.api.deps import (
    get_gpt4o_client,
    get_gpt5_client,
    get_gpt35_client,
    get_gpt41_client,
)
from app.be_core.logger import logger


class VannaCustom(PG_VectorStore, OpenAI_Chat):
    def __init__(self, llm_model="GPT4o", config=None):
        PG_VectorStore.__init__(self, config=config)

        if llm_model == "GPT41":
            client = get_gpt41_client()
        elif llm_model == "GPT4o":
            client = get_gpt4o_client()
        elif llm_model == "GPT35":
            client = get_gpt35_client()
        elif llm_model == "GPT5":
            client = get_gpt5_client()
        else:
            logger.debug(
                f"Unknown LLM model '{llm_model}' provided ot the training api is being used. Defaulting to GPT-4o."
            )
            client = get_gpt4o_client()

        OpenAI_Chat.__init__(self, client=client, config=config)
