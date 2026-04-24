from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

# Define the structure you want back
class ConceptExplanation(BaseModel):
    concept: str = Field(description="The name of the concept")
    summary: str = Field(description="A one sentence summary")
    difficulty: str = Field(description="Either 'beginner', 'intermediate', or 'advanced'")

# Bind the structure directly to the model
structured_model = model.with_structured_output(ConceptExplanation)

# Simple prompt — no need for format instructions anymore
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Explain the concept of {topic}.")
])

messages = template.invoke({"topic": "vector embeddings"})

# Call the structured model
parsed = structured_model.invoke(messages)

print(parsed)
print(type(parsed))
print(parsed.concept)
print(parsed.difficulty)