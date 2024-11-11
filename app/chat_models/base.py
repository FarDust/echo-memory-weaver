from typing import Type
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.messages import ChatMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel


def base_inference_chain(
    model_name: str = "gpt-4o-mini",
    structured_output: Type[BaseModel] | None = None,
    **kwargs,
) -> Runnable:
    def _default_model_invoke(inputs) -> ChatMessage:
        chat_model: BaseChatModel = ChatOpenAI(model=model_name, **kwargs)
        final_runnable: Runnable = chat_model
        if structured_output and hasattr(chat_model, "with_structured_output"):
            final_runnable = chat_model.with_structured_output(
                schema=structured_output,
                **kwargs,
            )
        return final_runnable.invoke(inputs)

    return RunnableLambda(lambda inputs: _default_model_invoke(inputs))
