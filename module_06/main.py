from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from fastapi.responses import StreamingResponse
import os

load_dotenv()

app = FastAPI()

model = ChatAnthropic(model="claude-haiku-4-5-20251001")
parser = StrOutputParser()

loader = TextLoader("sample.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)
embeddings = VoyageAIEmbeddings(
    voyage_api_key=os.getenv("VOYAGE_API_KEY"),
    model="voyage-3-lite"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer the question. If the answer is not in the context, say you don't know.\n\nContext: {context}"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = RunnableParallel(
    context=retriever | RunnableLambda(format_docs),
    question=RunnableLambda(lambda x: x)
) | prompt | model | parser

class QuestionRequest(BaseModel):
    question: str
    
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask")
def ask(request: QuestionRequest):
    result = chain.invoke(request.question)
    return {"answer": result}

@app.post("/ask/stream")
def ask_stream(request: QuestionRequest):
    def generate():
        for chunk in chain.stream(request.question):
            yield chunk
    return StreamingResponse(generate(), media_type="text/plain")
