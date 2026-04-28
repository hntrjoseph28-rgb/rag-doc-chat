from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

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

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together and return the result."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together and return the result."""
    return a + b

@tool
def search_document(query: str) -> str:
    """Search the document for information about AI, machine learning, deep learning, NLP, or RAG."""
    results = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in results)

tools = [multiply, add, search_document]

agent = create_agent(model, tools, system_prompt="You are a helpful assistant. Use tools to answer questions.", debug=True)

questions = [
    "What is deep learning and what is 7 multiplied by 3?",
]

for question in questions:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"Q: {question}")
    print(f"A: {result['messages'][-1].content}")
    print("---")