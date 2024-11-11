from functools import partial
import numbers
from operator import itemgetter
from typing import Any, Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import ChatMessage
from langchain_core.runnables import Runnable, RunnableParallel, RunnableLambda

from app.chat_models.base import base_inference_chain
from logging import getLogger, Logger

from app.prompts.notes.sentiments.evaluation import SENTIMENT_EVALUATION_TEMPLATE
from app.prompts.notes.sentiments.parser import SENTIMENT_PARSING_TEMPLATE
from app.tools.sentiments_parser.models import (
    SentimentAnalysisCall,
    SentimentAnalysisResult,
    SentimentsAnalysisCalls,
    AnalysisModels,
)


class SentimentsParserInput(BaseModel):
    """
    The input of the Sentiments Parser
    """

    text: str = Field(
        description="The text that would be parsed for sentiments extractions",
    )


class SentimentsParserOutput(BaseModel):
    """
    The output of the Sentiments Parser
    """

    sentiments: list[SentimentAnalysisResult] = Field(
        description="The list of sentiments extracted from the text and their scores",
    )


def average_calculator(
    input_type: Type[BaseModel],
) -> Runnable:
    """
    given a list of scores, calculate the average of the scores.

    Parameters
    ----------
    input_type : Type[BaseModel]
        The input type of the tool

    Returns
    -------
    BaseTool
        The tool to calculate the average of the scores
    """

    def _average_from_pydantic_scores(scores: dict[str, BaseModel]) -> BaseModel:
        """
        Calculate the average of the scores

        Parameters
        ----------
        scores : list[Model]
            The list of scores to be averaged

        Returns
        -------
        Model
            The average of the scores
        """
        model_dumps = [score.model_dump() for score in scores.values()]
        result: dict[str, Any] = {}
        for index, dump in enumerate(model_dumps):
            for key, value in dump.items():
                if isinstance(value, numbers.Number):
                    original_value = result.get(key, 0)
                    result[key] = (original_value * index + value) / (index + 1)
                elif isinstance(value, dict):
                    original_value = result.get(key, {})
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, numbers.Number):
                            original_sub_value = original_value.get(sub_key, 0)
                            original_value[sub_key] = (
                                original_sub_value * index + sub_value
                            ) / (index + 1)
                        else:
                            original_value[sub_key] = (
                                sub_value  # Todo: Change to frequency counter
                            )
                    result[key] = original_value
                else:
                    result[key] = value

        for key, value in result.items():
            if isinstance(value, float):
                result[key] = round(value, 3)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        value[sub_key] = round(sub_value, 3)
                    else:
                        value[sub_key] = sub_value
        return input_type.model_validate(result)

    return RunnableLambda(
        func=_average_from_pydantic_scores,
        name="Average Calculator",
    )


def naive_llm_sentiment_strength_evaluator(
    input_type: Type[BaseModel] = SentimentAnalysisCall,
    structured_output: Type[BaseModel] = SentimentAnalysisResult,
    executions: int = 5,
) -> BaseTool:
    """
    Tool to naively evaluate the sentiment strength of a text using
    an LLM

    Parameters
    ----------
    input_type : Type[BaseModel]
        The input type of the tool
    structured_output : Type[BaseModel]
        The structured output of the tool
    executions : int
        The number of executions to be done for the tool to obtain the average sentiment strength

    Returns
    -------
    BaseTool
        The tool to evaluate the sentiment strength of a text using an LLM
    """

    template = (
        RunnableLambda(
            lambda x: {
                **x,
                **{"structured_output": structured_output.model_json_schema()},
            }
        )
        | SENTIMENT_EVALUATION_TEMPLATE
    )

    base_chain = (
        template | base_inference_chain(structured_output=structured_output)
    ).as_tool(
        args_schema=input_type,
        name="Naive Sentiment Evaluator",
        description="Tool to naively evaluate the sentiment strength of a text using an LLM",
    )

    calculator = average_calculator(input_type=SentimentAnalysisResult)

    final_chain: Runnable = (
        RunnableParallel({str(index): base_chain for index in range(executions)})
        | calculator
    )
    return final_chain


class SentimentsParserTool(BaseTool):
    name: str = "Sentiments Parser"
    description: str = (
        "Tool to obtain a list of sentiments from a already processed text, "
        "that states clearly the sentiments of the text"
    )
    args_schema: Type[BaseModel] = SentimentsParserInput
    return_direct: bool = True
    prompt: ChatPromptTemplate = Field(
        description="The prompt to be used in the tool",
        default=RunnableLambda(
            lambda x: {
                **x,
                **{"structured_output": SentimentsAnalysisCalls.model_json_schema()},
            }
        )
        | SENTIMENT_PARSING_TEMPLATE,
    )
    inference_model: Runnable[str | list[ChatMessage] | PromptValue, ChatMessage] = (
        Field(
            description="The chat model to use for the tool",
            default_factory=partial(
                base_inference_chain, structured_output=SentimentsAnalysisCalls
            ),
        )
    )

    inference_tools: dict[AnalysisModels, BaseTool] = Field(
        description="The tools to be used for the inference",
        default_factory=lambda: {"llm": naive_llm_sentiment_strength_evaluator()},
    )

    _logger: Logger = getLogger(__name__)

    def route_analysis_calls(
        self, inputs: SentimentsAnalysisCalls
    ) -> SentimentsParserOutput:
        """Route the analysis call to the correct tool"""

        executions = []

        for analysis_call in inputs.categories:
            tool = self.inference_tools[analysis_call.analysis_tool]
            executions.append(tool.invoke(input=analysis_call.model_dump()))

        result = SentimentsParserOutput(sentiments=executions)

        return result

    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> SentimentsParserOutput:
        """Use the tool to extract the sentiments from the text"""

        analysis_calls_chain: Runnable = (
            {"text": itemgetter("text")}
            | self.prompt
            | self.inference_model
            | RunnableLambda(
                func=self.route_analysis_calls,
                name="Route Analysis Calls",
            )
        )
        return analysis_calls_chain.invoke(input={"text": text})
