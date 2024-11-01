from langchain_core.prompts import ChatPromptTemplate

SENTIMENT_EVALUATION_MESSAGES = [
    (
        "system",
        (
            """
            <PERSONALITY>
                You are an expert on evaluating the strength of a sentiment in a `TEXT` between 0 and 1, where 0 is the weakest and 1 is the strongest.
                You are allowed to infer other metrics about your analysis such as confidence that must be bound between 0 and, where 0 is the weakest and 1 is the strongest.
            </PERSONALITY>

            <OUTPUT_FORMAT>
                {structured_output}
            </OUTPUT_FORMAT>
            """
        )
    ),
    (
        "user",
        """
        <TASK>
            Please analyze the following `TEXT` and execute the evaluation over the `SENTIMENT` to determine the sentiment strength. Format the response as a JSON defined in the `OUTPUT_FORMAT` section.
        </TASK>

        <SENTIMENT>
            {sentiment}
        </SENTIMENT>

        <TEXT>
            {text}
        </TEXT>

        Formatted Output:
        """
    )
]

SENTIMENT_EVALUATION_TEMPLATE = ChatPromptTemplate.from_messages(SENTIMENT_EVALUATION_MESSAGES)
SENTIMENT_EVALUATION_TEMPLATE.name = "Sentiment Evaluation Template"