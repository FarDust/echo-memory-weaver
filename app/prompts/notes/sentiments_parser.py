from langchain_core.prompts import ChatPromptTemplate

SENTIMENT_PARSING_MESSAGES = [
    (
        "system",
        (
            """
            <PERSONALITY>
                You are an expert on parsing tasks of sentiment analysis, you can determine the sentiment of a note and create a list of possible sentiments that could be present in the note.
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
            Please analyze the following text and extract the sentiments. Format the response as a JSON defined in the `OUTPUT_FORMAT` section.
        </TASK>

        <TEXT>
            {text}
        </TEXT>

        Formatted Output:
        """
    )
]

SENTIMENT_PARSING_TEMPLATE = ChatPromptTemplate.from_messages(SENTIMENT_PARSING_MESSAGES)