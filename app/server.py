from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from langchain_openai.chat_models import ChatOpenAI
from langchain import hub
from pathlib import Path

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
        context=Path("assets/personal/gabriel.md").read_text(encoding="utf-8"),
    ),
    path="/notes",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
