from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5-20251001")
parser = StrOutputParser()

def retrieve_context(question):
    return "Japan is an island country in East Asia. Its capital is Tokyo, which has been the seat of government since 1868."

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer the question.\n\nContext: {context}"),
    ("human", "{question}")
])

pipeline = RunnableParallel(
    context=RunnableLambda(retrieve_context),
    question=RunnableLambda(lambda x: x)
) | prompt | model | parser

result = pipeline.invoke("What is the capital of Japan?")
print (result)
