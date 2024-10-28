from operator import itemgetter
from typing import Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import ChatMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnableSequence, RunnablePassthrough
from app.chat_models.base import base_inference_chain
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.embeddings.base import base_embedding_chain
from app.prompts.notes.sentiment import SENTIMENT_ANALYSIS_TEMPLATE
from app.prompts.notes.summarization import SUMMARIZATION_TEMPLATE


class NoteIngestInput(BaseModel):
    title: str = Field(
        description="The title of the note",
    )
    text: str = Field(
        description="The text to be ingested",
    )
    extra: str = Field(
        description="Extra information to be ingested",
        default="",
    )


class PromptTasks(BaseModel):
    summarization: ChatPromptTemplate = Field(
        description="The summarization prompt to be used",
    )
    sentiments_categorization: ChatPromptTemplate = Field(
        description="The sentiment categorization prompt to be used",
    )

class NoteIngestTool(BaseTool):
    name: str = "Note Ingest, Summarize & Categorization"
    description: str = (
        "useful for when you need to ingest a note, summarize it and categorize it"
    )
    args_schema: Type[BaseModel] = NoteIngestInput
    return_direct: bool = True
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
    prompt_tasks: PromptTasks = Field(
        description="The prompt to be used in a given task",
        default=PromptTasks(
            sentiments_categorization=SENTIMENT_ANALYSIS_TEMPLATE,
            summarization=SUMMARIZATION_TEMPLATE,
        )
    )
    text_formatter: PromptTemplate = Field(
        description="The template to be used for the prompt",
        default_factory=lambda: PromptTemplate.from_template("Title: {title}\n\nText: {text}\n\nExtra: {extra}\n\n")
    )

    def _run(
        self,
        title: str,
        text: str,
        extra: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""

        to_list: Runnable = RunnableLambda(lambda x: [x]) | {"texts": RunnablePassthrough() }
        compile_to_simple_text = RunnableLambda(
            lambda inputs: self.text_formatter.format(**inputs)
        )

        base_chain: Runnable = RunnableSequence(
            {
                "title": itemgetter("title"),
                "text": itemgetter("text"),
                "extra": itemgetter("extra"),
            },
            {
                "formatted_note": compile_to_simple_text,
                "context": RunnableLambda(lambda x: self.context),
            },
            {
                "formatted_note": itemgetter("formatted_note"),
                "sentiments": RunnablePassthrough()
                | self.prompt_tasks.sentiments_categorization
                | self.inference_model
                | StrOutputParser(),
                "summary": RunnablePassthrough()
                | self.prompt_tasks.summarization
                | self.inference_model
                | StrOutputParser(),
            },
            {
                "bare_ingest": itemgetter("formatted_note") | to_list 
                    | self.ingest_tool,
                "sentiments": {
                    "sentiments": itemgetter("sentiments"),
                    "embedding": itemgetter("sentiments")
                    | to_list
                    | self.ingest_tool,
                },
                "summary": {
                    "summary": itemgetter("summary"),
                    "embedding": itemgetter("summary") | to_list | self.ingest_tool,
                },
            },
            {
                "bare_ingest": itemgetter("bare_ingest"),
                "sentiments": itemgetter("sentiments"),
                "summary": itemgetter("summary"),
            },
            RunnableLambda(lambda x: x)
        )

        return base_chain.invoke({"title": title, "text": text, "extra": extra})
