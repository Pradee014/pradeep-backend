# Pradeep's Portfolio Backend

This is the backend API service for Pradeep's personal portfolio. It is built with **FastAPI** to provide data and services to the frontend application and future RAG (Retrieval-Augmented Generation) agents.

## 🚀 Tech Stack

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (Fast Python package installer)
- **Deployment Platform**: [Render](https://render.com/)
- **Frontend Host**: [Vercel](https://vercel.com/) (Connects to this API)
- **Database**: [Pinecone](https://www.pinecone.io/) (Planned for Vector/RAG operations)
- **RAG Capabilities**:
  - **Tools**: LinkedIn, Grok/Ollama Cloud, Gmail, Calendar
  - **Agent**: Integrated Chatbot for portfolio interaction

## 🛠️ Local Development Setup

Follow these steps to get the backend running locally on your machine.

### Prerequisites

- Python 3.11+ installed
- [uv](https://github.com/astral-sh/uv) installed (Recommended replacement for pip)

### 1. Clone & Navigate
```bash
# If you haven't already
cd pradeep-backend
```

### 2. Install Dependencies
This project uses `uv` for blazing fast dependency management.

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies from pyproject.toml
uv pip install -e .
```
*Alternatively, if you must use standard pip:*
```bash
pip install -e .
```

### 3. Run the Server
Start the development server with live reload:

```bash
uvicorn app.main:app --reload
```

The server will start at: `http://127.0.0.1:8000`

### 4. Verify
Open your browser or curl the health endpoint:
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **API Docs (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## ⚙️ Configuration

### CORS (Cross-Origin Resource Sharing)
The backend is configured to allow requests from specific frontend origins.
See `app/main.py` to update the `origins` list:

```python
origins = [
    "http://localhost:3000",             # Local frontend
    "https://your-vercel-app.vercel.app" # Production frontend (Update this!)
]
```

### Deployment
This project is configured for deployment on **Render**.
 Ensure looking into build commands if required (usually `uv sync` and start command `uvicorn app.main:app --host 0.0.0.0 --port 10000`).