from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings

def default_embedding(model_name: str = "text-embedding-ada-002", *args, **kwargs) -> Embeddings:
    embedding_model = OpenAIEmbeddings(model=model_name, *args, **kwargs)
    return embedding_model

def base_embedding_chain(
    model_name: str = "text-embedding-ada-002", *args, **kwargs
) -> Runnable[list[str], list[list[float]]]:
    def _default_model_invoke(inputs: list[str]) -> list[list[float]]:
        embedding_model = OpenAIEmbeddings(model=model_name, *args, **kwargs)
        return embedding_model.embed_documents(inputs)

    return RunnableLambda(lambda inputs: _default_model_invoke(inputs))
