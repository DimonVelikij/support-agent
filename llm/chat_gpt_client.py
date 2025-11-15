import http
import os
import logging

import requests
from dotenv import load_dotenv
from .base_llm import LLM

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChatGptClient(LLM):
    def __init__(self):
        self.__url = os.getenv("OPENAI_URL")
        self.__api_key = os.getenv("OPENAI_API_KEY")
        self.__proxy = os.getenv("OPENAI_PROXY")
        self.__model_name = os.getenv("OPENAI_MODEL_NAME")
        self.__temperature = float(os.getenv("OPENAI_TEMPERATURE"))

    def make_request(self, system_prompt: str, user_prompt: str) -> tuple[dict, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.__api_key}"
        }

        data = {
            "model": self.__model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.__temperature,
        }

        proxies = {
            'http': self.__proxy,
            'https': self.__proxy,
        }

        response = requests.request(
            http.HTTPMethod.POST,
            self.__url,
            headers=headers,
            json=data,
            proxies=proxies,
            timeout=(10, 10),
        )

        content = response.json()

        if (
            "choices" not in content or
            len(content["choices"]) == 0 or
            "message" not in content["choices"][0] or
            "content" not in content["choices"][0]["message"]
        ):
            raise Exception(f"некорректный ответ от LLM, отсутствуют данные в ответе: "
                            f"['choices'][0]['message']['content']")

        return content, content['choices'][0]['message']['content']


chat_gpt_client = ChatGptClient()
