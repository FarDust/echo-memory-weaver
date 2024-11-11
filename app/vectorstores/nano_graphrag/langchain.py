import asyncio
from dataclasses import dataclass, field
import json
from typing import Type
from nano_graphrag.base import BaseVectorStorage
from logging import getLogger, Logger
from langchain_core.vectorstores.base import VectorStore
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig

from app.embeddings.base import default_embedding


@dataclass
class LangChainVectorDBStorage(BaseVectorStorage):
    vector_store_class: Type[VectorStore] = Milvus
    vector_store_class_kwargs: dict = field(default_factory=dict)
    retriever_kwargs: dict = field(default_factory=dict)
    cosine_better_than_threshold: float = 0.2

    _logger: Logger = getLogger(__name__)

    def __post_init__(self):
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        vector_db_storage_cls_kwargs = self.global_config.get(
            "vector_db_storage_cls_kwargs", {}
        )
        self.vector_store_class = vector_db_storage_cls_kwargs["vector_store_class"]
        self.vector_store_class_kwargs = vector_db_storage_cls_kwargs.get(
            "vector_store_class_kwargs", {}
        )
        original_collection_name = self.vector_store_class_kwargs.get(
            "collection_name", "default"
        )
        self.vector_store_class_kwargs["collection_name"] = (
            f"{original_collection_name}_{self.namespace}"
        )
        self.vector_store = self.vector_store_class(embedding_function=default_embedding(), **self.vector_store_class_kwargs)

        self._retriever = ( RunnableLambda(
            lambda input: self.vector_store.similarity_search_with_score(
                **input,
            ),
            name="Get Similarity Search",
        ) | RunnableLambda(
            lambda results_documents: [
                {
                    **dp.metadata,
                    "content": dp.page_content,
                    "id": dp.id or dp.metadata["pk"],
                    "distance": distance
                }
                for dp, distance in results_documents
                if distance > self.cosine_better_than_threshold
            ],
            name="Format Results & Filter",
        )).with_config(
            config=RunnableConfig(
                run_name=f"Retriever: {self.namespace}",
                metadata=self.vector_store.as_retriever()._get_ls_params(),
                tags=[self.vector_store_class.__name__, self.vector_store.embeddings.__class__.__name__],
            )
        )

    async def upsert(self, data: dict[str, dict]):
        self._logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            self._logger.warning("You insert an empty data to vector DB")
            return []
        print(json.dumps(
            data,
            indent=4,
            default=str
        ))
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        documents = [
            Document(
                id=d["__id__"],
                page_content=contents[index],
                metadata={k: v for k, v in d.items() if k in self.meta_fields},
            )
            for index, d in enumerate(list_data)
        ]
        results = self.vector_store.add_documents(
            documents=documents,
            ids=[d["__id__"] for d in list_data],
        )
        return {
            "insert": results,
            "update": [],
        }

    async def query(self, query: str, top_k=5):
        results: list[dict] = await self._retriever.ainvoke(input={
            "query": query,
            "k": top_k
        })
        return results

    async def index_done_callback(self):
        pass
        # self._client.save()
