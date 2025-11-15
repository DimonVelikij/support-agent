from datetime import datetime
from pydantic import Field
from model.base_model import BaseAPIModel


class NewRequest(BaseAPIModel):
    query: str = Field(..., description="Запрос клиента")
    qdrant_result: list | None = Field(None, description="Извлеченные данные из векторонй базы")
    llm_result: dict | None = Field(None, description="Результат работы llm")
    feedback: bool | None = Field(None, description="Обратная связь по сформированному ответу")
    feedback_comment: str | None = Field(None, description="Комментарий к сформированному ответу")
    created_at: datetime | None = Field(None, description="Дата время создания записи")
    updated_at: datetime | None = Field(None, description="Дата время обновления записи")


class Request(NewRequest):
    id: int = Field(None, description="Идентификатор записи")
