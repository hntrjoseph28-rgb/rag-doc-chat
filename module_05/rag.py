from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()

loader = TextLoader("sample.txt")
docs = loader.load()

print (f"Number of documents: {len(docs)}")
print(f"Content preview: {docs[0].page_content[:200]}")


# --- Text Splitter ---

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_documents(docs)

print(f"\nNumber of chunks: {len(chunks)}")
# for i, chunk in enumerate(chunks):
#     print(f"\nChunk {i+1}:\n{chunk.page_content}")


# --- Embedding ---

from langchain_voyageai import VoyageAIEmbeddings
import os

embeddings = VoyageAIEmbeddings(
    voyage_api_key=os.getenv("VOYAGE_API_KEY"),
    model="voyage-3-lite"
)

sample_embedding = embeddings.embed_query("What is machine learning?")

print(f"\nEmbedding length: {len(sample_embedding)}")
print(f"First 5 numbers: {sample_embedding[:5]}")


# --- Vector Store ---

from langchain_community.vectorstores import FAISS

vectorstore= FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

results = retriever.invoke("What is machine learning?")

for result in results:
    print(result.page_content)
    print("---")
