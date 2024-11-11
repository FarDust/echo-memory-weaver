from langchain_core.prompts import ChatPromptTemplate

SUMMARIZATION_MESSAGES = [
    (
        "system",
        (
            """
            <PERSONALITY>
                You are an specialist on summarize the content of a note in key points and analyze objectivity of the content of the note.
                The person that write the note is described in `CONTEXT` section.
            </PERSONALITY>

            <CONTEXT>
                {context}
            </CONTEXT>

            <RESTRICTIONS>
                - The general summary should be separated from the key points by a `---` line.
                - You prefer to summarize the content of the text in key points.
                - Add [BEGIN HIGHLY OBJECTIVE][END HIGHLY OBJECTIVE] to the key points that are highly objective.
                - Add [BEGIN HIGHLY SUBJECTIVE][END HIGHLY SUBJECTIVE] to the key points that are highly subjective.
                - Add [BEGIN NEUTRAL][END NEUTRAL] to the key points that are neutral.
            </RESTRICTIONS>

            <EXAMPLE>
                <INPUT>
                    Title: Remember

                    Text: Nadie a quien le guste el poder debería tenerlo

                    Extra: 
                </INPUT>

                <OUTPUT>
                    ## General Summary
                    Expresses a sentiment that individuals who enjoy power should not hold it, suggesting a cautionary view on the nature of power and authority.

                    ---

                    ## Key Points
                    - [BEGIN SUBJECTIVE] The author believes that power should not be in the hands of those who enjoy it. [END SUBJECTIVE]
                    - [BEGIN NEUTRAL] The statement reflects a philosophical perspective on power dynamics. [END NEUTRAL]

                    ---

                    ### Reasoning
                    1. Presents a subjective opinion regarding the moral implications of power.
                    2. The statement can be interpreted as a critique of power structures and the motivations behind individuals seeking power.
                    3. The content does not provide objective evidence or examples, making it heavily reliant on personal belief.

                    ### Conclusion
                    Presents a subjective viewpoint on power, emphasizing that those who desire it should be wary of its possession, while also reflecting a broader philosophical stance on authority.
                </OUTPUT>
            </EXAMPLE>
            """
        ),
    ),
    (
        "user",
        """
        <TASK>
            Summarize the `NOTE` and create a list of key points. Analyze the objectivity of the content of the note.
            The output format should be markdown including a general summary of the note and the key points.

            The person that write the note is described in `CONTEXT` section.

                - Think step by step before responding.
                - Provide a reasoning list with no more than 3 bullet points.
                - Conclude with a final response after the reasoning.
                - Do not repeat information unnecessarily.

                --- End the response after the final conclusion ---
        </TASK>

        <CONTEXT>
            {context}
        </CONTEXT>

        <NOTE>
            {formatted_note}
        </NOTE>

        Markdown Response:
        """,
    ),
]

SUMMARIZATION_TEMPLATE = ChatPromptTemplate.from_messages(SUMMARIZATION_MESSAGES)
SUMMARIZATION_TEMPLATE.name = "Summarization Template"
