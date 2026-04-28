from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
import os
import shutil

load_dotenv()

app = FastAPI(title="rag-doc-chat")

model = ChatAnthropic(model="claude-haiku-4-5-20251001")
parser = StrOutputParser()
embeddings = VoyageAIEmbeddings(
    voyage_api_key=os.getenv("VOYAGE_API_KEY"),
    model="voyage-3-lite"
)

vectorstore = None
retriever = None
session_store = {}

def load_documents(path: str):
    global vectorstore, retriever
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(path)
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(path)
    elif ext in [".md", ".txt"]:
        loader = TextLoader(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use ONLY the context below to answer the question. If the answer is not in the context, say you don't know.\n\nContext: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

load_documents("docs/sample.txt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

def make_chain():
    def get_context(inputs):
        docs = retriever.invoke(inputs["question"])
        return format_docs(docs)

    from langchain_core.runnables import RunnablePassthrough
    
    chain = (
        RunnablePassthrough.assign(context=lambda x: get_context(x))
        | prompt
        | model
        | parser
    )
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"

@app.get("/health")
def health_check():
    return {"status": "ok", "document_loaded": retriever is not None}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = [".txt", ".pdf", ".docx", ".md"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return {"error": f"Unsupported file type: {ext}"}
    path = f"docs/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    load_documents(path)
    return {"status": "ok", "filename": file.filename}

@app.post("/load-gdoc")
def load_gdoc(url: str):
    from langchain_community.document_loaders import GoogleDriveLoader
    try:
        from langchain_community.document_loaders import UnstructuredURLLoader
        loader = UnstructuredURLLoader(urls=[url])
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        global vectorstore, retriever
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        return {"status": "ok", "url": url}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask/stream")
def ask_stream(request: QuestionRequest):
    chain = make_chain()
    def generate():
        for chunk in chain.stream(
            {"question": request.question},
            config={"configurable": {"session_id": request.session_id}}
        ):
            yield chunk
    return StreamingResponse(generate(), media_type="text/plain")