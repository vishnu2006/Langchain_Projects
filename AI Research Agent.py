import os
from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

from langchain_community.utilities import SerpAPIWrapper

load_dotenv()

app = FastAPI()

# -----------------------------
# LLM + Embeddings
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embeddings = OpenAIEmbeddings()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

vector_store = None

# -----------------------------
# Web Search Tool
# -----------------------------
search = SerpAPIWrapper()

web_tool = Tool(
    name="Web Search",
    func=search.run,
    description="Use when answer is not found in documents"
)

# -----------------------------
# Upload + Index Docs
# -----------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store

    path = f"./{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embeddings)

    return {"message": "Document indexed successfully 🚀"}

# -----------------------------
# Ask Agent
# -----------------------------
@app.post("/ask")
def ask_question(question: str = Form(...)):
    global vector_store

    tools = []

    if vector_store:
        retriever = vector_store.as_retriever()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        doc_tool = Tool(
            name="Document QA",
            func=lambda q: qa_chain.invoke({"query": q}),
            description="Use for document-based questions"
        )

        tools.append(doc_tool)

    tools.append(web_tool)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )

    response = agent.run(question)

    return {
        "answer": response,
        "mode": "doc + web + memory enabled"
    }

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {"status": "🔥 LangChain AI Research Agent running"}
