from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnableLambda
import os
import time

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5-20251001")
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use ONLY the context below to answer the question. If the answer is not in the context, you MUST say 'I don't know based on the provided context.' Do not use any outside knowledge.\n\nContext: {context}"),
    ("human", "{question}")
])

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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = RunnableParallel(
    context=retriever | RunnableLambda(format_docs),
    question=RunnableLambda(lambda x: x)
) | prompt | model | parser

eval_dataset = [
    {"question": "What is machine learning?", "expected": "subset of AI"},
    {"question": "What is deep learning?", "expected": "neural networks"},
    {"question": "What is the capital of France?", "expected": "don't know"},
]

print("\n--- Eval Results ---")
passed = 0
for item in eval_dataset:
    time.sleep(20)
    try:
        result = rag_chain.invoke(item["question"])
        success = item["expected"].lower() in result.lower()
        status = "PASS" if success else "FAIL"
        if success:
            passed += 1
    except Exception as e:
        status = "ERROR"
        print(f"{status} | Q: {item['question']} | {str(e)[:50]}")
        continue
    print(f"{status} | Q: {item['question']}")

print(f"\n{passed}/{len(eval_dataset)} tests passed")