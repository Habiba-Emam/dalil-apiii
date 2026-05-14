# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import os
import json
import uuid
import shutil

from langchain_community.document_loaders import DirectoryLoader, CSVLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# =========================
# Paths
# =========================

BASE_PATH = os.path.join(os.getcwd(), "data")
SOURCE_DATA_PATH = os.path.join(BASE_PATH, "Source_data")
VECTOR_DB_PATH = os.path.join(BASE_PATH, "vector_db_index")

# =========================
# Models
# =========================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1
)

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)

# =========================
# Vector Store
# =========================

def get_or_create_vectorstore():
    if os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
        return FAISS.load_local(
            VECTOR_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    all_docs = []

    if os.path.exists(SOURCE_DATA_PATH):
        all_docs.extend(
            DirectoryLoader(
                SOURCE_DATA_PATH,
                glob="*.pdf",
                loader_cls=PyPDFLoader
            ).load()
        )

        all_docs.extend(
            DirectoryLoader(
                SOURCE_DATA_PATH,
                glob="*.csv",
                loader_cls=CSVLoader,
                loader_kwargs={"encoding": "utf-8"}
            ).load()
        )

        all_docs.extend(
            DirectoryLoader(
                SOURCE_DATA_PATH,
                glob="*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            ).load()
        )

    if not all_docs:
        all_docs = [
            Document(
                page_content="Initial empty university knowledge base.",
                metadata={"source": "system"}
            )
        ]

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    ).split_documents(all_docs)

    vectorstore = FAISS.from_documents(chunks, embedding_model)

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    vectorstore.save_local(VECTOR_DB_PATH)

    return vectorstore


vectorstore = get_or_create_vectorstore()


def save_vectorstore():
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    vectorstore.save_local(VECTOR_DB_PATH)


def search_knowledge_base(query: str, k: int = 5):
    docs = vectorstore.similarity_search(query, k=k)

    sources = []
    context_parts = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content

        context_parts.append(content)

        sources.append({
            "source": source,
            "content": content[:300]
        })

    return "\n\n".join(context_parts), sources


# =========================
# Schemas
# =========================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    userRole: str = "user"
    chatHistory: List[ChatMessage] = []


class VectorUpsertRequest(BaseModel):
    documentId: Optional[str] = None
    collection: str
    content: str
    metadata: Optional[Dict[str, Any]] = {}


class VectorDeleteRequest(BaseModel):
    documentId: str


# =========================
# Prompts
# =========================

def format_chat_history(chat_history: List[ChatMessage]) -> str:
    if not chat_history:
        return "No previous chat history."

    formatted = []

    for msg in chat_history[-10:]:
        formatted.append(f"{msg.role}: {msg.content}")

    return "\n".join(formatted)


ADMIN_DETECTION_PROMPT = """
You are an admin action detector for a university AI system.

Your job:
Detect if the admin wants to create, update, delete, or get university data.

Return ONLY valid JSON.

Allowed response types:

1. If this is a normal question:
{
  "type": "answer"
}

2. If this is an admin action:
{
  "type": "admin_action",
  "action": "create | update | delete | get",
  "collection": "students | professors | courses | policies | performance | general",
  "filter": {},
  "data": {},
  "reply": "short explanation"
}

3. If missing important info:
{
  "type": "clarification",
  "reply": "Ask for the missing info",
  "missingFields": []
}

User role: {user_role}
Message: {message}
"""

ANSWER_PROMPT = """
You are Dalil AI, an expert university assistant.

You answer questions about university information, students, professors, courses,
policies, performance, and general academic questions.

Use the provided university context when relevant.
Use chat history to understand follow-up questions.

If the answer is not found in the context, say that the information is not available
in the current university knowledge base.

Chat History:
{chat_history}

University Context:
{context}

User Question:
{question}

Return a helpful direct answer.
"""


# =========================
# FastAPI App
# =========================

app = FastAPI(title="Dalil AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Endpoints
# =========================

@app.get("/")
def health_check():
    return {
        "status": "Dalil AI is online",
        "service": "university-ai-agent"
    }


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        # 1. Admin action detection
        if req.userRole == "admin":
            admin_prompt = ADMIN_DETECTION_PROMPT.format(
                user_role=req.userRole,
                message=req.message
            )

            admin_result = llm.invoke(admin_prompt).content.strip()

            try:
                parsed_admin = json.loads(admin_result)

                if parsed_admin.get("type") in ["admin_action", "clarification"]:
                    return parsed_admin

            except json.JSONDecodeError:
                pass

        # 2. Normal RAG answer
        context, sources = search_knowledge_base(req.message)

        chat_history_text = format_chat_history(req.chatHistory)

        answer_prompt = ANSWER_PROMPT.format(
            chat_history=chat_history_text,
            context=context,
            question=req.message
        )

        answer = llm.invoke(answer_prompt).content.strip()

        return {
            "type": "answer",
            "reply": answer,
            "toolUsed": "rag",
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vector/upsert")
def vector_upsert(req: VectorUpsertRequest):
    try:
        document_id = req.documentId or str(uuid.uuid4())

        metadata = req.metadata or {}
        metadata["documentId"] = document_id
        metadata["collection"] = req.collection

        doc = Document(
            page_content=req.content,
            metadata=metadata
        )

        vectorstore.add_documents([doc])
        save_vectorstore()

        return {
            "message": "Vector document upserted successfully",
            "documentId": document_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/vector/delete")
def vector_delete(req: VectorDeleteRequest):
    # FAISS delete by metadata is not simple in LangChain FAISS.
    # Recommended production solution:
    # use Qdrant, Pinecone, Chroma, or rebuild FAISS after delete.
    return {
        "message": "FAISS delete by documentId is not directly supported here.",
        "recommendation": "Use /vector/rebuild after deleting from MongoDB, or switch to Qdrant/Pinecone for real delete support.",
        "documentId": req.documentId
    }


@app.post("/vector/rebuild")
def vector_rebuild():
    try:
        global vectorstore

        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH)

        vectorstore = get_or_create_vectorstore()

        return {
            "message": "Vector database rebuilt successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))