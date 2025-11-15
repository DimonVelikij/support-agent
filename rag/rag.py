import os
import logging

from typing import Optional
from dotenv import load_dotenv
from commands.confluence_qdrant import ConfluenceToQdrant
from llm.base_llm import LLM
from database.postgresql_client import PostgresqlClient
from utils.prompt import compile_prompt
from model.base_model import BaseAPIModel
from model.request import Request, NewRequest
from pydantic import Field

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RagResponse(BaseAPIModel):
    error: str | None = Field(None, description="Текст ошибки")
    context: list[dict] | None = Field(None, description="Контекст из векторной базы данных")
    llm_response: dict | None = Field(None, description="Данные ответа от LLM")
    answer: str | None = Field(None, description="Окончательный текст ответа на запрос клиента")
    request: Request | None = Field(None, description="Объект запроса клиента")


class RagPipeline:
    def __init__(self, llm: LLM, postgresql_client: PostgresqlClient):
        logger.info("Инициализация pipeline")

        self.__confluence_url = os.getenv('CONFLUENCE_URL')
        self.__confluence_username = os.getenv('CONFLUENCE_USERNAME')
        self.__confluence_api_token = os.getenv('CONFLUENCE_API_TOKEN')

        self.__qdrant_url = os.getenv('QDRANT_HOST')
        self.__qdrant_port = int(os.getenv('QDRANT_PORT'))
        self.__use_local_qdrant = os.getenv('USE_LOCAL_QDRANT', 'True').lower() == 'true'
        self.__collection_name = os.getenv('COLLECTION_NAME')

        self.__embedding_model = os.getenv('EMBEDDING_MODEL')

        self._score_threshold = float(os.getenv('SCORE_THRESHOLD'))

        self.__confluence_to_qdrant = ConfluenceToQdrant(
            confluence_url=self.__confluence_url,
            confluence_username=self.__confluence_username,
            confluence_api_token=self.__confluence_api_token,
            qdrant_url=self.__qdrant_url,
            qdrant_port=int(self.__qdrant_port),
            use_local_qdrant=self.__use_local_qdrant,
            collection_name=self.__collection_name,
            embedding_model=self.__embedding_model,
        )
        self.__llm = llm
        self.__postgresql_client = postgresql_client
        self.__system_prompt = compile_prompt(os.getenv('SYSTEM_PROMPT_TEMPLATE_PATH'))
        self.__user_prompt_path = os.getenv('USER_PROMPT_TEMPLATE_PATH')

        logger.info("Pipeline готов к работе")

    def run(self, query: str, top_k: int = 5, filter_space: Optional[str] = None,
            score_threshold: float = 0.5) -> RagResponse:
        logger.info(f"Поиск по запросу в векторной базе данных: {query}, по пространству: {filter_space}")

        try:
            new_request = NewRequest(
                query=query,
            )
            request = self.__postgresql_client.create_request(new_request)

            context = self.__confluence_to_qdrant.search(
                query=query,
                top_k=top_k,
                filter_space=filter_space,
                score_threshold=score_threshold,
            )

            if len(context) == 0:
                return RagResponse(
                    error="По вашему запросу ничего не удалось найти",
                )

            logger.info(f"Найдено результатов в векторной базе данных: {len(context)}")

            user_prompt = compile_prompt(self.__user_prompt_path, {
                'query': query,
                'context': context,
            })

            llm_response, answer = self.__llm.make_request(self.__system_prompt, user_prompt)

            logger.info(f"Ответ от llm получен")

            request.qdrant_result = context
            request.llm_result = llm_response

            request = self.__postgresql_client.update_request(request)

            return RagResponse(
                context=context,
                llm_response=llm_response,
                answer=answer,
                request=request,
            )
        except Exception as err:
            logger.error(f"Ошибка поиска: {err}")
            raise err
