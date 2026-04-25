from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer in one sentence."),
    ("human", "{question}")
])

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"question": "What is the capital of Japan?"})
print("\n", result)

# prompt_output = prompt.invoke({"question": "What is the capital of Japan?"})
# print(type(prompt_output))
# print("\n", prompt_output)


# --- RunnableLambda ---

from langchain_core.runnables import RunnableLambda

def make_uppercase(text):
    return text.upper()

uppercase_chain = prompt | model | parser | RunnableLambda(make_uppercase)

result2 = uppercase_chain.invoke({"question": "What is the capital of Japan?"})
print("\n", result2)

