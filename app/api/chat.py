from typing import List, Dict, Any, TypedDict, Literal, Union, Optional
from fastapi import APIRouter
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END
from app.core.config import settings
from fastapi.responses import StreamingResponse
from ai_sdk.types import CoreSystemMessage, CoreUserMessage, CoreAssistantMessage

import logging
import sys

# --- Logging Setup ---
# Create a logger specific to this module
logger = logging.getLogger("chat_api")
logger.setLevel(logging.INFO)

# Create formatters and handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File Handler - logs to chat_api.log
file_handler = logging.FileHandler("chat_api.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console Handler - logs to stdout
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


router = APIRouter()

# --- 1. Define State ---
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    context: str

# --- 2. Initialize Models & Tools ---
embeddings = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL,
    dimensions=settings.EMBEDDING_DIMENSIONS,
    api_key=settings.OPENAI_API_KEY.get_secret_value()
)

vector_store = PineconeVectorStore(
    index_name=settings.PINECONE_INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=settings.PINECONE_API_KEY.get_secret_value()
)

# llm = ChatOpenAI(
#     base_url=settings.OLLAMA_BASE_URL,
#     model=settings.OLLAMA_MODEL if hasattr(settings, "OLLAMA_MODEL") else "llama3", # Default or from settings
#     api_key="ollama", # Ollama doesn't check key, but library requires one
#     streaming=True
# )

# Initialize AI SDK Provider (for streaming response)
# REMOVED: Replaced by native ollama.AsyncClient inside endpoint
# llm_streaming = openai(
#     model=settings.OLLAMA_MODEL if hasattr(settings, "OLLAMA_MODEL") else "llama3",
#     api_key="ollama",
#     base_url=settings.OLLAMA_BASE_URL
# )

# --- 3. Define Nodes ---

def retrieve(state: AgentState):
    """Retrieve relevant documents from Pinecone."""
    # Logic needs to handle the message format change if we were parsing raw messages here,
    # but currently we extract text in the endpoint and pass it to state as a simple list of dicts or similar.
    # The 'last_message' here assumes state["messages"] has a standard format.
    # We will ensure state["messages"] is compatible or adjust logic.
    last_message = state["messages"][-1]["content"] 
    docs = vector_store.similarity_search(last_message, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    return {"context": context}

def generate(state: AgentState):
    """Generate response using Ollama."""
    return state

# --- 4. Build Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app_graph = workflow.compile()

# --- 5. Data Models ---
class TextPart(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: str  # 'user', 'assistant', or 'system'
    content: Optional[str] = None
    parts: Optional[List[TextPart]] = None  # Handles the new multimodal parts array

class ChatRequest(BaseModel):
    messages: List[Message]

# --- 6. Endpoint ---

# 6. Endpoint
@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 0. Log Incoming Request
    logger.info(f"--- Incoming Chat Request ---")
    logger.info(f"Messages count: {len(request.messages)}")
    logger.info(f"Last user message raw: {request.messages[-1]}")

    # 1. Extract extraction logic
    last_message = request.messages[-1]
    user_input = ""
    
    if last_message.parts:
        user_input = "".join([p.text for p in last_message.parts if p.type == "text"])
    elif last_message.content:
        user_input = last_message.content
        
    logger.info(f"Extracted User Input: {user_input}")
    
    # 2. Run LangGraph to get Context
    graph_messages = [{"role": "user", "content": user_input}]
    initial_state = {"messages": graph_messages, "context": ""}
    final_state = await app_graph.ainvoke(initial_state)
    context = final_state["context"]
    
    logger.info(f"Retrieved Context Length: {len(context)} chars")
    
    # 3. Inject context into system prompt
    system_prompt = f"You are a helpful assistant. Use the following context to answer the user's question:\n\nContext:\n{context}"
    
    # 4. Prepare messages for AI SDK
    # AI SDK expects a list of CoreMessage objects.
    ai_messages = []
    
    # Add System Prompt first
    ai_messages.append(CoreSystemMessage(content=system_prompt))
    
    for m in request.messages:
        content_text = ""
        if m.parts:
            content_text = "".join([p.text for p in m.parts if p.type == "text"])
        elif m.content:
            content_text = m.content
        
        if m.role == "user":
            ai_messages.append(CoreUserMessage(content=content_text))
        elif m.role == "assistant":
            ai_messages.append(CoreAssistantMessage(content=content_text))
        # Add other roles if needed, or default/ignore

    # 5. Initialize Custom AI SDK Provider
    from ai_sdk import stream_text
    from app.lib.custom_ollama_provider import CustomOllamaProvider
    from app.lib.utils import to_data_stream_response
    
    # Use native Ollama API via our custom adapter
    base_url = settings.OLLAMA_BASE_URL
    model_name = settings.OLLAMA_MODEL if hasattr(settings, "OLLAMA_MODEL") else "llama3"
    
    # Prepare Auth Headers if API key exists
    headers = {}
    if settings.OLLAMA_API_KEY:
        key_val = settings.OLLAMA_API_KEY.get_secret_value()
        # Usually Bearer token for cloud or just basic auth? 
        # For Ollama Cloud, it's typically Basic auth or Bearer. 
        # Let's try Bearer first or just passing it if the library supports it.
        # But 'ollama.com' usually implies typical cloud auth.
        headers["Authorization"] = f"Bearer {key_val}"

    logger.info("Starting AI SDK Stream (Custom Native Provider)...")
    logger.info(f"Host: {base_url}")
    logger.info(f"Model: {model_name}")

    try:
        # 6. Stream Response
        # Note: stream_text is synchronous in this SDK version but returns an async iterator in result.text_stream
        result = stream_text(
            model=CustomOllamaProvider(
                model=model_name, 
                host=base_url,
                headers=headers
            ),
            messages=ai_messages
        )
        
        # 7. Return Streaming Response (Automated Data Stream Protocol with Custom Helper)
        return to_data_stream_response(result)
        
    except Exception as e:
        logger.error(f"Streaming Error: {str(e)}", exc_info=True)
        # Return a 500 or allow FastAPI to handle it, but logging is key.
        raise e
