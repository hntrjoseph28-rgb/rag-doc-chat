from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Define a template with a variable called {topic}
template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in two sentences."
)

# Fill in the variable
prompt = template.invoke({"topic": "vector databases"})

print(prompt)
print(type(prompt))


# --- ChatPromptTemplate ---

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains technical concepts simply."),
    ("human", "Explain {topic} in two sentences.")
])

# Fill in the variable
messages = chat_template.invoke({"topic": "vector databases"})

print(messages)


# --- Connecting the model ---

from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

# Fill in the template
messages = chat_template.invoke({"topic": "Docker containers"})
response = model. invoke(messages)

print("\n", response.content)


# --- String Output Parsers ---

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# Pass the AIMessage through the parser
response = model.invoke(messages)
clean_output = parser.invoke(response)

print("\n", clean_output)
print(type(clean_output))
