import json
import os
import psycopg2
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

from model.request import NewRequest, Request

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PostgresqlClient:
    def __init__(self):
        self.__host = os.getenv("DB_HOST")
        self.__port = int(os.getenv("DB_PORT"))
        self.__user = os.getenv("DB_USER")
        self.__password = os.getenv("DB_PASSWORD")
        self.__dbname = os.getenv("DB_NAME")
        self.__connection = None
        self.__cursor = None

    def connect(self):
        try:
            self.__connection = psycopg2.connect(
                dbname=self.__dbname,
                user=self.__user,
                password=self.__password,
                host=self.__host,
                port=self.__port,
            )
            self.__cursor = self.__connection.cursor()
            logger.info(f"соединение с базой данных установлено")
        except Exception as err:
            logger.error(f"ошибка подключения к базе данных: {str(err)}")
            raise err

    def disconnect(self):
        if self.__cursor:
            self.__cursor.close()
        if self.__connection:
            self.__connection.close()
        logger.info(f"соединение с базой данных закрыто")

    def reconnect(self):
        self.disconnect()
        self.connect()

    def is_active_connection(self):
        return self.__connection is not None and self.__connection.close == 0

    def find_request(self, request_id: str) -> Request | None:
        if not self.is_active_connection():
            logger.info(f"соединение с базой данных неактивно")
            self.reconnect()

        query = f"""
            SELECT 
                id,
                query,
                qdrant_result,
                llm_result,
                feedback,
                feedback_comment,
                created_at,
                updated_at 
            FROM requests
            WHERE id = %s
        """

        params = (request_id,)

        try:
            self.__cursor.execute(query, params)
            self.__connection.commit()
            result = self.__cursor.fetchone()

            if result is None:
                return None

            return Request(
                id=result[0],
                query=result[1],
                qdrant_result=result[2],
                llm_result=result[3],
                feedback=result[4],
                feedback_comment=result[5],
                created_at=result[6],
                updated_at=result[7],
            )
        except Exception as err:
            logger.error(f"во время выполнения запроса на получение заявки по id возникли ошибки: {str(err)}")
            raise err

    def create_request(self, new_request: NewRequest) -> Request:
        if not self.is_active_connection():
            logger.info(f"соединение с базой данных неактивно")
            self.reconnect()

        time_now = datetime.now(timezone.utc)

        query = f"""
            INSERT INTO requests (
                query,
                qdrant_result,
                llm_result,
                feedback,
                feedback_comment,
                created_at,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s) 
            RETURNING id
        """

        params = (
            new_request.query,
            json.dumps(new_request.qdrant_result) if new_request.qdrant_result is not None else '{}',
            json.dumps(new_request.llm_result) if new_request.llm_result is not None else '{}',
            new_request.feedback,
            new_request.feedback_comment,
            time_now,
            time_now,
        )

        try:
            self.__cursor.execute(query, params)
            self.__connection.commit()
            insert_id = self.__cursor.fetchone()[0]

            return Request(
                id=insert_id,
                query=new_request.query,
                qdrant_result=new_request.qdrant_result,
                llm_result=new_request.llm_result,
                feedback=new_request.feedback,
                feedback_comment=new_request.feedback_comment,
                created_at=time_now,
                updated_at=time_now,
            )
        except Exception as err:
            logger.error(f"во время выполнения запроса на создание заявки возникли ошибки: {str(err)}")
            raise err

    def update_request(self, request: Request) -> Request:
        if not self.is_active_connection():
            logger.info(f"соединение с базой данных неактивно")
            self.reconnect()

        time_now = datetime.now(timezone.utc)

        query = f"""
            UPDATE requests 
            SET qdrant_result = %s,
                llm_result = %s,
                feedback = %s,
                feedback_comment = %s,
                updated_at = %s
            WHERE id = %s
        """

        params = (
            json.dumps(request.qdrant_result) if request.qdrant_result is not None else '{}',
            json.dumps(request.llm_result) if request.llm_result is not None else '{}',
            request.feedback,
            request.feedback_comment,
            time_now,
            request.id,
        )

        try:
            self.__cursor.execute(query, params)
            self.__connection.commit()

            request.updated_at = time_now

            return request
        except Exception as err:
            logger.error(f"во время выполнения запроса на обновление заявки возникли ошибки: {str(err)}")
            raise err


postgresql_client = PostgresqlClient()
