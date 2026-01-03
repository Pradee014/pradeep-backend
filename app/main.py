from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

app = FastAPI()

# Allow any localhost port for development
origins = [
    "http://localhost:3000",
    settings.FRONTEND_URL, 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex="https?://localhost:\d+", # Allow any localhost port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ADD THIS ROUTE ---
@app.get("/")
def read_root():
    return {"message": "Zero-Cost Enterprise RAG is live!"}

@app.get("/health")
def health_check():
    return {"status": "awake", "message": "Backend is ready"}

from app.api import chat
app.include_router(chat.router, prefix="/api")
