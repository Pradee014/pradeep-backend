from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your specific Vercel frontend to access the API
origins = [
    "http://localhost:3000",
    "https://pradeep-frontend-six.vercel.app/" # <--- Remember to update this later!
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
