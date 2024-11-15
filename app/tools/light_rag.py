from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from pathlib import Path
import numpy as np

from typing import Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import ChatMessage
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

from app.embeddings.base import base_embedding_chain
from app.chat_models.base import base_inference_chain
from logging import getLogger, Logger


class LightRAGInput(BaseModel):
    texts: list[str] = Field(
        description="The text to be ingested",
    )


class LightRAGInterface(BaseTool):
    name: str = "LightRAG"
    description: str = "useful to store text data a understand it better with a graph"
    args_schema: Type[BaseModel] = LightRAGInput
    return_direct: bool = True
    user: str = Field(
        description="The user to be used in the tool",
    )
    context: str = Field(
        description="The context to be used in the tool to known the user",
        default="",
    )
    inference_model: Runnable[str | list[ChatMessage] | PromptValue, ChatMessage] = (
        Field(
            description="The chat model to use for the tool",
            default_factory=base_inference_chain,
        )
    )
    ingest_tool: Runnable[list[str], list[list[float]]] = Field(
        description="The embedding model to use for the tool",
        default_factory=base_embedding_chain,
    )

    user_prompt: str = Field(
        description="The prompt to be used to intervene the user query",
        default="{query}",
    )

    insert_mode: bool = Field(
        description="The mode to be used to intervene the user query",
        default=True,
    )

    _logger: Logger = getLogger(__name__)

    def inference_model_langchain_adapter(
        self,
    ):
        async def _adapter(
            prompt,
            system_prompt=None,
            history_messages=[],
            **kwargs,
        ) -> str:
            messages: list[ChatMessage] = []
            if system_prompt:
                messages.append(ChatMessage(role="system", content=system_prompt))
            messages.extend(
                [
                    ChatMessage(
                        role=message["role"],
                        content=message["content"],
                    )
                    for message in history_messages
                ]
            )
            messages.append(
                ChatMessage(role="user", content=self.user_prompt.format(query=prompt))
            )

            completion_chain: Runnable = self.inference_model | StrOutputParser()

            return await completion_chain.ainvoke(input=messages)

        return _adapter

    async def embedding_adapter(self, texts: list[str]) -> np.ndarray:
        embeddings = await self.ingest_tool.ainvoke(input=texts)
        return np.array(embeddings)

    def _run(
        self,
        texts: list[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> list[str]:
        """Use the tool synchronously to ingest user data."""

        db_path = Path(f"./light_rag/{self.user}").absolute().resolve()
        db_path.mkdir(parents=True, exist_ok=True)

        rag = LightRAG(
            working_dir=db_path,
            llm_model_func=self.inference_model_langchain_adapter(),
            embedding_func=EmbeddingFunc(
                embedding_dim=1536, max_token_size=819, func=self.embedding_adapter
            ),
        )

        return_texts = ["Inserted"]

        if self.insert_mode:
            rag.insert(texts)
        else:
            return_texts = [
                rag.query(text, param=QueryParam(mode="global")) for text in texts
            ]
        return return_texts
