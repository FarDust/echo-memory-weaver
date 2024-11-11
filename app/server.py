from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from langchain_openai.chat_models import ChatOpenAI
from langchain import hub
from pathlib import Path

from app.tools.sentiments_parser.definition import SentimentsParserTool
from app.tools.nano_graph_rag import NanoGraphRAGInterface
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
        ingest_tool=NanoGraphRAGInterface(
            user="gabriel",
        ),
    ),
    path="/notes",
)

add_routes(
    app,
    NanoGraphRAGInterface(
        user="gabriel",
    ),
    path="/search",
)

add_routes(
    app,
    SentimentsParserTool(),
    path="/sentiments",
)

# Mark timestamp on the chain to tag user evolution and allow override for batch past events
# Get Sentiments from a text
# Get summaries for text
# Create GraphRag Communities for Summaries? maybe use the whole text makes more sense? But maybe no depending on the token length?
# Create GraphRag Communities for Sentiments
# Run Sentiments strength analysis over extracted Sentiments index them into somewhere. Where is the best place to store this?
# Create GraphRag of topics of interest for the user? Maybe query the Summaries and Sentiments and create a GraphRag of the topics of interest for the user
# Add marks for private and public data given structured categories. What categorization is the best for this? Something like very private (For sexual orientation, health, etc), private (For personal data), inner circle (For mentioned friends, family, etc), outer circle (For data that the user is ok to share with not so close people and family), public (For data that the user is ok to share with everyone)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
