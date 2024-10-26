from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.messages import ChatMessage
from langchain_openai.chat_models import ChatOpenAI


def base_inference_chain(model_name: str = "gpt-4o-mini", *args, **kwargs) -> Runnable:
    def _default_model_invoke(inputs) -> ChatMessage:
        chat_model = ChatOpenAI(model=model_name, *args, **kwargs)
        return chat_model.invoke(inputs)

    return RunnableLambda(lambda inputs: _default_model_invoke(inputs))
