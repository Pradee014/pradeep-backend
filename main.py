from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your specific Vercel frontend to access the API
origins = [
    "http://localhost:3000",  # For local testing
    "https://portfolio-frontend-abc.vercel.app" # <--- REPLACE with your actual Vercel URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "awake", "message": "Backend is ready"}
