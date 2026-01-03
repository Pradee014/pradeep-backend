# Pradeep's Portfolio Backend

This is the backend API service for Pradeep's personal portfolio. It is built with **FastAPI** to provide data and services to the frontend application and future RAG (Retrieval-Augmented Generation) agents.

## 🚀 Tech Stack

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Package Manager**: pip (Standard Python package installer)
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


### 1. Clone & Navigate
```bash
# If you haven't already
cd pradeep-backend
```

### 2. Install Dependencies
Create a virtual environment and install the required packages:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies from requirements.txt
pip install -r requirements.txt
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
### Deployment & Infrastructure Status
- **Frontend**: Deployed on Vercel (Separate Repository).
- **Backend**: Deployed on **Render (Free Tier)**.
  - **Constraint**: **512 MB RAM Limit**. All integrations must be memory-efficient.
  - **Build Command**: `pip install -r requirements.txt`
  - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port 10000`
- **Database**: Pinecone setup complete.


# 🤖 Personal Executive AI Assistant

A state-of-the-art **Agentic RAG** (Retrieval-Augmented Generation) system designed to act as a digital surrogate for professional interactions. This assistant manages your professional persona by answering queries based on your Resume and GitHub, while autonomously handling scheduling and communications.

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | Next.js (Vercel) |
| **Backend** | FastAPI (Render) |
| **The Brain** | Ollama Cloud (GPT-OSS / DeepSeek) |
| **Vector DB** | Pinecone (512-Dimensions) |
| **Orchestrator** | LangGraph (Stateful Agents) |
| **Protocols** | Model Context Protocol (MCP) |

---

## 🧠 System Architecture



### 1. High-Efficiency Retrieval (RAG)
To optimize for the **Pinecone Free Tier**, this project utilizes **Matryoshka Embeddings**. Data from my Resume and GitHub READMEs are encoded using OpenAI's `text-embedding-3-small` and truncated to **512 dimensions**. This maintains ~99% retrieval accuracy while significantly reducing the memory footprint.

### 2. Stateful Agent Logic
Instead of a simple linear chain, the backend uses **LangGraph** to create a cyclic state machine. The "Brain" (Ollama Cloud) evaluates user intent and decides whether to:
* **Retrieve:** Search the Pinecone index for technical or professional background.
* **Act:** Interact with external tools via MCP.
* **Respond:** Synthesize information into a human-like, professional response.

### 3. Agentic Tools (MCP Integration)
The assistant is equipped with functional tools to bridge the gap between conversation and action:
* **📅 Google Calendar:** Real-time availability checks and meeting scheduling.
* **🐙 GitHub API:** Dynamic retrieval of repository logic and project summaries.
* **✉️ Communication:** Automated drafting for LinkedIn and Email outreach.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* OpenAI & Pinecone API Keys
* Ollama Cloud Access
* Google Cloud Console (Calendar API enabled)

### Installation
1. Clone the repo: `git clone https://github.com/your-username/your-repo.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env`:
   ```env
   OPENAI_API_KEY=your_key
   PINECONE_API_KEY=your_key
   OLLAMA_CLOUD_API_KEY=your_key