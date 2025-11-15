from abc import ABC, abstractmethod


class LLM(ABC):

    @abstractmethod
    def make_request(self, system_prompt: str, user_prompt: str) -> tuple[dict, str]:
        raise NotImplementedError
