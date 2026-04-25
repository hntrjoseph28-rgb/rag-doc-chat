from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5-20251001")
parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer in one short sentence."),
    ("human", "{question}")
])

formal_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer in a formal and detailed way."),
    ("human", "{question}")
])

parallel_chain = RunnableParallel(
    short=short_prompt | model | parser,
    formal=formal_prompt | model | parser
)

result = parallel_chain.invoke({"question": "What is the capital of Japan?"})
print("Short:", result["short"])
print("Formal:", result["formal"])
