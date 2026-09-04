# 🌐 Web URL Knowledge Bot (RAG with Streamlit, PostgreSQL & Gemini)

A Retrieval-Augmented Generation (RAG) chatbot built using **Python**, **Streamlit**, **PostgreSQL (pgvector)**, and **Google Gemini**.

Scrape content from any website URL, split it into semantic chunks, generate embeddings using Google's `text-embedding-004`, index vectors into PostgreSQL with pgvector, and ask questions with contextual grounded answers powered by Google Gemini.

---

## 🚀 Features

- **🌐 Web Scraper & Text Chunker:** Automatically fetches webpage content, strips boilerplate/navigation, and generates overlapping chunks.
- **🧠 Google Gemini Integration:**
  - Embeddings generation with `models/text-embedding-004` (768 dimensions).
  - Contextual question answering with `gemini-1.5-flash`, `gemini-1.5-pro`, or `gemini-2.0-flash`.
- **🐘 PostgreSQL + pgvector Storage:** Stores vectorized document chunks with cosine similarity HNSW indexing for rapid retrieval.
- **💬 Streamlit Chat Interface:**
  - Interactive chat session with streaming capability and history.
  - Source inspection expander displaying retrieved chunks and cosine similarity scores.
  - Filter questions across all indexed websites or focus on a single indexed URL.
  - Knowledge base management (view chunk counts, remove URLs, or clear database).

---

## 📋 Prerequisites

1. **Python 3.9+**
2. **Local PostgreSQL DB Server** (running locally on `localhost:5432`):
   - Works directly with standard local PostgreSQL server installations (e.g. PostgreSQL for Windows/macOS/Linux).
   - If `pgvector` extension is available, it automatically uses native `pgvector` acceleration with HNSW indexing; if not installed, it automatically uses standard PostgreSQL vector array mode with zero extra setup needed.
   - Alternatively, you can run PostgreSQL via Docker:
     ```bash
     docker run -d --name postgres-pgvector -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
     ```
3. **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/).

---

## 🛠️ Installation

1. **Clone the repository and navigate to the folder:**
   ```bash
   cd streamlit-calculator
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file from the `.env.example` template:
   ```env
   GOOGLE_API_KEY=your_google_gemini_api_key_here
   PGHOST=localhost
   PGPORT=5432
   PGDATABASE=postgres
   PGUSER=postgres
   PGPASSWORD=your_postgres_password
   PGSSLMODE=disable
   ```
   *(Note: You can also enter or override these settings directly in the Streamlit sidebar UI).*

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 💡 How to Use

1. **Configure Settings (Sidebar):**
   - Provide your **Google Gemini API Key**.
   - Enter your **PostgreSQL Connection Details** and click **🔌 Test & Initialize Database**.
2. **Ingest Web Content:**
   - Enter any public webpage URL (e.g. `https://en.wikipedia.org/wiki/Artificial_intelligence`).
   - Click **🚀 Scrape & Store Embeddings**. The application will fetch the page, chunk it, embed the text, and store it in PostgreSQL.
3. **Chat & Retrieve:**
   - Ask any question in the chat input.
   - The bot will perform vector similarity search against PostgreSQL and generate a grounded response citing sources!
   - Expand the **🔎 View Retrieved Context Sources** section below any response to inspect matching passages and similarity scores.