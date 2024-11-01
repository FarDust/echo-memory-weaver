from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from langchain_openai.chat_models import ChatOpenAI
from langchain import hub
from pathlib import Path

from app.tools.light_rag import LightRAGInterface
from app.tools.sentiments_parser.definition import SentimentsParserTool
from app.tools.text_ingestor import NoteIngestTool


app = FastAPI()


class SampleInputs(BaseModel):
    question: str


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


def sample_chain() -> Runnable:
    prompt = hub.pull("smithing-gold/assumption-checker")
    return prompt | ChatOpenAI(name="gpt-4o-mini")

base_context = Path("assets/personal/gabriel.md").read_text(encoding="utf-8")

# Edit this to add the chain you want to add
add_routes(
    app,
    sample_chain().with_types(
        input_type=SampleInputs,
    ),
    path="/hello",
)

add_routes(
    app,
    NoteIngestTool(
        context=base_context,
        ingest_tool=LightRAGInterface(
            user="gabriel",
            context=base_context,
        ),
    ),
    path="/notes",
)

add_routes(
    app,
    LightRAGInterface(
        user="gabriel",
        insert_mode=False,
    ),
    path="/global",
)

add_routes(
    app,
    SentimentsParserTool(),
    path="/sentiments",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
