"""
Загрузка данных из Confluence в Qdrant с векторным поиском
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from atlassian import Confluence
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from typing import List, Dict, Optional
from datetime import datetime
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfluenceToQdrant:
    """
    Класс для загрузки данных из Confluence в Qdrant
    с поддержкой векторного поиска
    """

    def __init__(
        self,
        confluence_url: str,
        confluence_username: str,
        confluence_api_token: str,
        qdrant_url: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "confluence_docs",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        use_local_qdrant: bool = True
    ):
        # Confluence клиент
        self.confluence = Confluence(
            url=confluence_url,
            username=confluence_username,
            password=confluence_api_token,
            cloud=False
        )

        # Qdrant клиент
        if use_local_qdrant:
            # Локальный Qdrant
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                port=qdrant_port
            )
        else:
            # Облачный Qdrant или с аутентификацией
            self.qdrant_client = QdrantClient(
                url=f"https://{qdrant_url}",
                api_key=qdrant_api_key
            )

        self.collection_name = collection_name

        # Модель для эмбеддингов
        logger.info(f"Загрузка модели эмбеддингов: {embedding_model}")
        self.model = SentenceTransformer(embedding_model, device='cpu')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Размерность векторов: {self.embedding_dim}")

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def create_collection(self, recreate: bool = False):
        """
        Создание коллекции в Qdrant

        Args:
            recreate: Если True, удалит существующую коллекцию и создаст новую
        """
        try:
            # Проверяем существование коллекции
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if collection_exists:
                if recreate:
                    logger.info(f"Удаление существующей коллекции {self.collection_name}")
                    self.qdrant_client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Коллекция {self.collection_name} уже существует")
                    return

            # Создаём коллекцию
            logger.info(f"Создание коллекции {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE  # Косинусное расстояние
                )
            )

            # Создаём индексы для фильтрации
            logger.info("Создание индексов для быстрой фильтрации...")
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="page_id",
                field_schema="keyword"
            )
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="space_key",
                field_schema="keyword"
            )

            logger.info(f"✅ Коллекция {self.collection_name} успешно создана")

        except Exception as e:
            logger.error(f"❌ Ошибка создания коллекции: {e}")
            raise

    def get_page_content(self, page_id: str) -> Optional[Dict]:
        """Получение содержимого страницы Confluence"""
        try:
            page = self.confluence.get_page_by_id(
                page_id=page_id,
                expand='body.storage,version,space'
            )

            html_content = page['body']['storage']['value']
            soup = BeautifulSoup(html_content, 'html.parser')

            # Удаляем ненужные элементы
            for element in soup(["script", "style", "meta", "link"]):
                element.decompose()

            text = soup.get_text(separator='\n', strip=True)

            return {
                'page_id': page_id,
                'title': page['title'],
                'text': text,
                'url': f"{self.confluence.url}/pages/viewpage.action?pageId={page_id}",
                'space': page['space']['key'],
                'version': page['version']['number']
            }
        except Exception as e:
            logger.error(f"❌ Ошибка получения страницы {page_id}: {e}")
            return None

    def chunk_text(self, text: str) -> List[str]:
        """Разбивка текста на чанки"""
        return self.text_splitter.split_text(text)

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Создание векторных представлений"""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def store_in_qdrant(self, page_data: Dict, chunks: List[str], embeddings: List[List[float]]):
        """
        Сохранение данных в Qdrant

        Каждый чанк сохраняется как отдельная точка с метаданными
        """
        points = []

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    'page_id': page_data['page_id'],
                    'page_title': page_data['title'],
                    'chunk_text': chunk,
                    'chunk_index': idx,
                    'url': page_data['url'],
                    'space_key': page_data['space'],
                    'version': page_data['version'],
                    'created_at': datetime.now().isoformat(),
                    'total_chunks': len(chunks)
                }
            )
            points.append(point)

        # Batch upload в Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"✅ Сохранено {len(points)} чанков в Qdrant")

    def delete_page_chunks(self, page_id: str):
        """
        Удаление всех чанков страницы
        Полезно для обновления страницы
        """
        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="page_id",
                            match=MatchValue(value=page_id)
                        )
                    ]
                )
            )
            logger.info(f"✅ Удалены чанки страницы {page_id}")
        except Exception as e:
            logger.error(f"❌ Ошибка удаления чанков: {e}")

    def process_page(self, page_id: str, update: bool = False):
        """
        Полный процесс обработки одной страницы

        Args:
            page_id: ID страницы Confluence
            update: Если True, сначала удалит существующие чанки
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📄 Обработка страницы {page_id}")
        logger.info(f"{'='*60}")

        # Если это обновление, удаляем старые чанки
        if update:
            self.delete_page_chunks(page_id)

        # 1. Получаем контент
        page_data = self.get_page_content(page_id)
        if not page_data:
            logger.error(f"❌ Не удалось получить данные страницы {page_id}")
            return False

        logger.info(f"✓ Получена страница: {page_data['title']}")
        logger.info(f"  Пространство: {page_data['space']}")
        logger.info(f"  Версия: {page_data['version']}")

        # 2. Разбиваем на чанки
        chunks = self.chunk_text(page_data['text'])
        logger.info(f"✓ Создано {len(chunks)} чанков")

        # 3. Создаем эмбеддинги
        logger.info(f"⏳ Создание эмбеддингов...")
        embeddings = self.create_embeddings(chunks)
        logger.info(f"✓ Создано {len(embeddings)} эмбеддингов")

        # 4. Сохраняем в Qdrant
        logger.info(f"⏳ Сохранение в Qdrant...")
        self.store_in_qdrant(page_data, chunks, embeddings)

        logger.info(f"✅ Страница {page_id} успешно обработана")
        return True

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_page_id: Optional[str] = None,
        filter_space: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Семантический поиск в Qdrant

        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            filter_page_id: Фильтр по ID страницы
            filter_space: Фильтр по пространству Confluence
            score_threshold: Минимальный порог релевантности (0-1)

        Returns:
            Список результатов с метаданными и оценками релевантности
        """
        # Создаём эмбеддинг для запроса
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0].tolist()

        # Формируем фильтры
        filter_conditions = []
        if filter_page_id:
            filter_conditions.append(
                FieldCondition(
                    key="page_id",
                    match=MatchValue(value=filter_page_id)
                )
            )
        if filter_space:
            filter_conditions.append(
                FieldCondition(
                    key="space_key",
                    match=MatchValue(value=filter_space)
                )
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        # Выполняем поиск
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold
        )

        # Форматируем результаты
        results = []
        for hit in search_result:
            results.append({
                'id': hit.id,
                'score': hit.score,
                'page_id': hit.payload['page_id'],
                'page_title': hit.payload['page_title'],
                'chunk_text': hit.payload['chunk_text'],
                'chunk_index': hit.payload['chunk_index'],
                'url': hit.payload['url'],
                'space_key': hit.payload['space_key'],
                'total_chunks': hit.payload.get('total_chunks', 0)
            })

        logger.info(f"✅ Найдено {len(results)} результатов")
        return results

    def get_statistics(self) -> Dict:
        """Получение статистики коллекции"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)

            # Получаем уникальные page_id через scroll
            unique_pages = set()
            unique_spaces = set()

            # Используем scroll для получения всех точек
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )

            for point in scroll_result[0]:
                unique_pages.add(point.payload['page_id'])
                unique_spaces.add(point.payload['space_key'])

            return {
                'total_chunks': collection_info.points_count,
                'total_pages': len(unique_pages),
                'total_spaces': len(unique_spaces),
                'vector_dimension': self.embedding_dim,
                'collection_name': self.collection_name
            }
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {}

    def get_all_child_pages(self, page_id: str) -> List[str]:
        """Получение всех дочерних страниц рекурсивно"""
        all_pages = [page_id]

        try:
            children = self.confluence.get_page_child_by_type(
                page_id=page_id,
                type='page',
                start=0,
                limit=100
            )

            for child in children:
                child_id = child['id']
                all_pages.extend(self.get_all_child_pages(child_id))

        except Exception as e:
            logger.warning(f"⚠ Предупреждение при получении дочерних страниц: {e}")

        return all_pages

    def process_page_tree(self, root_page_id: str, update: bool = False):
        """
        Обработка дерева страниц

        Args:
            root_page_id: ID корневой страницы
            update: Если True, обновит существующие страницы
        """
        logger.info(f"🌳 Получение дерева страниц от {root_page_id}...")
        all_page_ids = self.get_all_child_pages(root_page_id)

        logger.info(f"📊 Найдено {len(all_page_ids)} страниц для обработки\n")

        success_count = 0
        fail_count = 0

        for idx, page_id in enumerate(all_page_ids, 1):
            logger.info(f"\n[{idx}/{len(all_page_ids)}]")
            try:
                if self.process_page(page_id, update=update):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(f"❌ Ошибка при обработке страницы {page_id}: {e}")
                fail_count += 1
                continue

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Обработка завершена!")
        logger.info(f"   Успешно: {success_count}")
        logger.info(f"   Ошибок: {fail_count}")
        logger.info(f"{'='*60}")


# Пример использования
if __name__ == "__main__":
    # ID страницы
    PAGE_ID = os.getenv('PAGE_ID')

    # Создаём экземпляр
    pipeline = ConfluenceToQdrant(
        confluence_url=os.getenv('CONFLUENCE_URL'),
        confluence_username=os.getenv('CONFLUENCE_USERNAME'),
        confluence_api_token=os.getenv('CONFLUENCE_API_TOKEN'),
        qdrant_url=os.getenv('QDRANT_HOST'),
        qdrant_port=int(os.getenv('QDRANT_PORT')),
        qdrant_api_key=os.getenv('QDRANT_API_KEY'),
        collection_name="confluence_docs",
        embedding_model=os.getenv('EMBEDDING_MODEL'),
        use_local_qdrant=os.getenv('USE_LOCAL_QDRANT') == 'True'
    )

    # Создаём коллекцию (только при первом запуске!)
    pipeline.create_collection(recreate=False)

    if os.getenv('LOAD_PAGE_TREE') == 'True':
        # обрабатываем всё дерево страниц
        pipeline.process_page_tree(PAGE_ID)
    else:
        # Обрабатываем одну страницу
        pipeline.process_page(PAGE_ID)

    # Пример поиска
    logger.info(f"\n{'='*60}")
    logger.info("🔍 Тестовый векторный поиск")
    logger.info(f"{'='*60}\n")

    results = pipeline.search(
        query="векторный индекс",
        top_k=3,
        score_threshold=0.5  # Только результаты с релевантностью > 50%
    )

    for idx, result in enumerate(results, 1):
        print(f"\n{idx}. Релевантность: {result['score']:.3f}")
        print(f"   Страница: {result['page_title']}")
        print(f"   URL: {result['url']}")
        print(f"   Чанк #{result['chunk_index']}")
        print(f"   Текст: {result['chunk_text'][:200]}...")
