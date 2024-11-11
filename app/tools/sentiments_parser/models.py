from pydantic import BaseModel, Field
from typing import Literal, Union

# Define individual models for each category with fixed emotion lists and descriptions


class AcceptingOpen(BaseModel):
    category: Literal["Accepting / Open"] = Field(
        "Accepting / Open",
        description="A state of acceptance and calmness, reflecting openness to experience.",
    )
    emotions: list[
        Literal[
            "Calm",
            "Centered",
            "Content",
            "Fulfilled",
            "Patient",
            "Peaceful",
            "Present",
            "Relaxed",
            "Serene",
            "Trusting",
        ]
    ] = Field(
        description="List of emotions associated with a state of acceptance and openness."
    )


class AlivenessJoy(BaseModel):
    category: Literal["Aliveness / Joy"] = Field(
        "Aliveness / Joy",
        description="A state of heightened energy and happiness, often associated with joy and enthusiasm.",
    )
    emotions: list[
        Literal[
            "Amazed",
            "Awe",
            "Bliss",
            "Delighted",
            "Eager",
            "Ecstatic",
            "Enchanted",
            "Energized",
            "Engaged",
            "Enthusiastic",
            "Excited",
            "Free",
            "Happy",
            "Inspired",
            "Invigorated",
            "Lively",
            "Passionate",
            "Playful",
            "Radiant",
            "Refreshed",
            "Rejuvenated",
            "Renewed",
            "Satisfied",
            "Thrilled",
            "Vibrant",
        ]
    ] = Field(description="List of emotions associated with joy and aliveness.")


class Angry(BaseModel):
    category: Literal["Angry"] = Field(
        "Angry",
        description="A state of frustration or irritation, often resulting in anger or aggression.",
    )
    emotions: list[
        Literal[
            "Annoyed",
            "Agitated",
            "Aggravated",
            "Bitter",
            "Contempt",
            "Cynical",
            "Disdain",
            "Disgruntled",
            "Disturbed",
            "Edgy",
            "Exasperated",
            "Frustrated",
            "Furious",
            "Grouchy",
            "Hostile",
            "Impatient",
            "Irritated",
            "Irate",
            "Moody",
            "On edge",
            "Outraged",
            "Pissed",
            "Resentful",
            "Upset",
            "Vindictive",
        ]
    ] = Field(description="List of emotions associated with anger or irritation.")


class Courageous(BaseModel):
    category: Literal["Courageous"] = Field(
        "Courageous",
        description="A state of bravery and confidence, reflecting strength and determination.",
    )
    emotions: list[
        Literal[
            "Powerful",
            "Adventurous",
            "Brave",
            "Capable",
            "Confident",
            "Daring",
            "Determined",
            "Free",
            "Grounded",
            "Proud",
            "Strong",
            "Worthy",
            "Valiant",
        ]
    ] = Field(description="List of emotions associated with courage and confidence.")


class Connected(BaseModel):
    category: Literal["Connected"] = Field(
        "Connected",
        description="A state of closeness and empathy, feeling connected to others or oneself.",
    )
    emotions: list[
        Literal[
            "Loving",
            "Accepting",
            "Affectionate",
            "Caring",
            "Compassion",
            "Empathy",
            "Fulfilled",
            "Present",
            "Safe",
            "Warm",
            "Worthy",
            "Curious",
            "Engaged",
            "Exploring",
            "Fascinated",
            "Interested",
            "Intrigued",
            "Involved",
            "Stimulated",
        ]
    ] = Field(description="List of emotions associated with connectedness and empathy.")


class DespairSad(BaseModel):
    category: Literal["Despair / Sad"] = Field(
        "Despair / Sad",
        description="A state of sorrow or hopelessness, often involving sadness or grief.",
    )
    emotions: list[
        Literal[
            "Anguish",
            "Depressed",
            "Despondent",
            "Disappointed",
            "Discouraged",
            "Forlorn",
            "Gloomy",
            "Grief",
            "Heartbroken",
            "Hopeless",
            "Lonely",
            "Longing",
            "Melancholy",
            "Sorrow",
            "Teary",
            "Unhappy",
            "Upset",
            "Weary",
            "Yearning",
        ]
    ] = Field(description="List of emotions associated with sadness or despair.")


class DisconnectedNumb(BaseModel):
    category: Literal["Disconnected / Numb"] = Field(
        "Disconnected / Numb",
        description="A state of detachment or numbness, often involving emotional distance or apathy.",
    )
    emotions: list[
        Literal[
            "Aloof",
            "Bored",
            "Confused",
            "Distant",
            "Empty",
            "Indifferent",
            "Isolated",
            "Lethargic",
            "Listless",
            "Removed",
            "Resistant",
            "Shut Down",
            "Uneasy",
            "Withdrawn",
        ]
    ] = Field(description="List of emotions associated with numbness or disconnection.")


class EmbarrassedShame(BaseModel):
    category: Literal["Embarrassed / Shame"] = Field(
        "Embarrassed / Shame",
        description="A state of self-consciousness or shame, often resulting in embarrassment or guilt.",
    )
    emotions: list[
        Literal[
            "Ashamed",
            "Humiliated",
            "Inhibited",
            "Mortified",
            "Self-conscious",
            "Useless",
            "Weak",
            "Worthless",
        ]
    ] = Field(description="List of emotions associated with shame or embarrassment.")


class Fear(BaseModel):
    category: Literal["Fear"] = Field(
        "Fear",
        description="A state of apprehension or dread, often involving worry or anxiety.",
    )
    emotions: list[
        Literal[
            "Afraid",
            "Anxious",
            "Apprehensive",
            "Frightened",
            "Hesitant",
            "Nervous",
            "Panic",
            "Paralyzed",
            "Scared",
            "Terrified",
            "Worried",
        ]
    ] = Field(description="List of emotions associated with fear or anxiety.")


class Fragile(BaseModel):
    category: Literal["Fragile"] = Field(
        "Fragile",
        description="A sensitive or vulnerable state, often involving feelings of helplessness.",
    )
    emotions: list[Literal["Helpless", "Sensitive"]] = Field(
        description="List of emotions associated with feeling fragile or vulnerable."
    )


class Grateful(BaseModel):
    category: Literal["Grateful"] = Field(
        "Grateful",
        description="A state of appreciation or thankfulness, often involving feelings of fortune or blessing.",
    )
    emotions: list[
        Literal[
            "Appreciative",
            "Blessed",
            "Delighted",
            "Fortunate",
            "Grace",
            "Humbled",
            "Lucky",
            "Moved",
            "Thankful",
            "Touched",
        ]
    ] = Field(description="List of emotions associated with gratitude or appreciation.")


class Guilt(BaseModel):
    category: Literal["Guilt"] = Field(
        "Guilt",
        description="A state of remorse or regret, often involving feelings of responsibility for harm caused.",
    )
    emotions: list[Literal["Regret", "Remorseful", "Sorry"]] = Field(
        description="List of emotions associated with guilt or remorse."
    )


class Hopeful(BaseModel):
    category: Literal["Hopeful"] = Field(
        "Hopeful",
        description="A state of optimism or encouragement, often involving expectation or trust.",
    )
    emotions: list[Literal["Encouraged", "Expectant", "Optimistic", "Trusting"]] = (
        Field(description="List of emotions associated with hope and optimism.")
    )


class Powerless(BaseModel):
    category: Literal["Powerless"] = Field(
        "Powerless",
        description="A state of helplessness or resignation, often involving feelings of entrapment.",
    )
    emotions: list[
        Literal["Impotent", "Incapable", "Resigned", "Trapped", "Victim"]
    ] = Field(description="List of emotions associated with feeling powerless.")


class Tender(BaseModel):
    category: Literal["Tender"] = Field(
        "Tender",
        description="A gentle or caring state, often involving feelings of vulnerability or warmth.",
    )
    emotions: list[
        Literal[
            "Calm",
            "Caring",
            "Loving",
            "Reflective",
            "Self-loving",
            "Serene",
            "Vulnerable",
            "Warm",
        ]
    ] = Field(description="List of emotions associated with tenderness and warmth.")


class Stressed(BaseModel):
    category: Literal["Stressed"] = Field(
        "Stressed",
        description="A state of strain or tension, often involving feelings of anxiety or fatigue.",
    )
    emotions: list[
        Literal[
            "Tense",
            "Anxious",
            "Burned out",
            "Cranky",
            "Depleted",
            "Edgy",
            "Exhausted",
            "Frazzled",
            "Overwhelm",
            "Rattled",
            "Rejecting",
            "Restless",
            "Shaken",
            "Tight",
            "Weary",
            "Worn out",
        ]
    ] = Field(description="List of emotions associated with stress or fatigue.")


class UnsettledDoubt(BaseModel):
    category: Literal["Unsettled / Doubt"] = Field(
        "Unsettled / Doubt",
        description="A state of uncertainty or doubt, often involving feelings of skepticism or hesitation.",
    )
    emotions: list[
        Literal[
            "Apprehensive",
            "Concerned",
            "Dissatisfied",
            "Disturbed",
            "Grouchy",
            "Hesitant",
            "Inhibited",
            "Perplexed",
            "Questioning",
            "Rejecting",
            "Reluctant",
            "Shocked",
            "Skeptical",
            "Suspicious",
            "Ungrounded",
            "Unsure",
            "Worried",
        ]
    ] = Field(
        description="List of emotions associated with feeling unsettled or doubtful."
    )


# Define a type alias for all sentiment categories
SentimentCategoryType = Union[
    AcceptingOpen,
    AlivenessJoy,
    Angry,
    Courageous,
    Connected,
    DespairSad,
    DisconnectedNumb,
    EmbarrassedShame,
    Fear,
    Fragile,
    Grateful,
    Guilt,
    Hopeful,
    Powerless,
    Tender,
    Stressed,
    UnsettledDoubt,
]

AnalysisModels = Literal["llm"]


class SentimentAnalysisCall(BaseModel):
    """
    A call to the sentiment analysis tool.
    """

    text: str = Field(
        description="The text to be analyzed",
    )
    sentiment: SentimentCategoryType = Field(
        description="The sentiment category to be analyzed",
    )
    analysis_tool: AnalysisModels = Field(
        description="The analysis model to be used, could be LLM, APIs, local models, etc...",
    )


class SentimentAnalysisResult(BaseModel):
    """
    The result of a sentiment analysis call.
    """

    sentiment: SentimentCategoryType = Field(
        description="The sentiment category that was analyzed"
    )
    score: float = Field(description="The overall sentiment score for the text")
    sub_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Optional breakdown of scores for different emotional intensities or confidence scores",
    )


class ExtendedSentimentAnalysisResult(SentimentAnalysisResult, BaseModel):
    """
    The Result of the sentiment analysis reconstructed with the original text
    """

    text: str = Field(
        description="The text to analyzed",
    )
    sentiment: SentimentCategoryType = Field(
        description="The sentiment category that was analyzed"
    )
    score: float = Field(description="The overall sentiment score for the text")
    sub_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Optional breakdown of scores for different emotional intensities or confidence scores",
    )


class SentimentsAnalysisCalls(BaseModel):
    """
    A list of sentiment categories to be analyzed.
    """

    categories: list[SentimentAnalysisCall] = Field(
        description="The list of sentiment categories to be analyzed",
    )
