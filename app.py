import os
from typing import Optional
import streamlit as st
from dotenv import load_dotenv
import tempfile

from db import VectorDatabase
from gemini_service import GeminiService
from scraper import extract_text_from_url, chunk_text
from document_processor import extract_text_from_document, chunk_document_text

# Load environment variables from .env file if available
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot: Web URL to PostgreSQL with Gemini",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "db_connected" not in st.session_state:
        st.session_state.db_connected = False


init_session_state()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Gemini API Key Section
    with st.expander("🔑 Google Gemini Settings", expanded=True):
        default_api_key = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))
        gemini_api_key = st.text_input(
            "Gemini API Key",
            value=default_api_key,
            type="password",
            help="Get your API key from Google AI Studio (https://aistudio.google.com/)",
        )
        gemini_chat_model = st.selectbox(
            "Chat Model",
            options=["gemini-3.8-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
            index=0,
        )
        embedding_model = st.text_input(
            "Embedding Model",
            value="gemini-embedding-2",
            disabled=True,
            help="Standard 768-dimensional text embedding model for Gemini.",
        )

    # Local PostgreSQL Database Section
    with st.expander("🐘 Local PostgreSQL Server Settings", expanded=True):
        st.caption("Connect to your local PostgreSQL instance (e.g., localhost:5432)")
        pg_host = st.text_input("Host", value=os.getenv("PGHOST", "localhost"), help="Local database host (e.g., localhost or 127.0.0.1)")
        pg_port = st.number_input("Port", value=int(os.getenv("PGPORT", "5432")), step=1, help="Default PostgreSQL port is 5432")
        pg_db = st.text_input("Database Name", value=os.getenv("PGDATABASE", "postgres"), help="Local database name")
        pg_user = st.text_input("User", value=os.getenv("PGUSER", "postgres"), help="Local database username")
        pg_pass = st.text_input("Password", value=os.getenv("PGPASSWORD", ""), type="password", help="Password for your local PostgreSQL user")
        pg_ssl = st.selectbox(
            "SSL Mode",
            options=["disable", "prefer", "require", "allow", "verify-ca", "verify-full"],
            index=0,
            help="Choose 'disable' or 'prefer' for local database servers",
        )

        db_instance = None
        if pg_host and pg_db and pg_user:
            db_instance = VectorDatabase(
                host=pg_host,
                port=int(pg_port),
                database=pg_db,
                user=pg_user,
                password=pg_pass,
                sslmode=pg_ssl,
            )

        if st.button("🔌 Test & Initialize Local Database", use_container_width=True):
            if db_instance:
                with st.spinner("Connecting to local PostgreSQL and initializing schema..."):
                    success, message = db_instance.test_connection()
                    if success:
                        try:
                            db_instance.init_db(dimension=768)
                            st.session_state.db_connected = True
                            st.success(f"✅ {message}")
                        except Exception as e:
                            st.error(f"❌ Initialization error: {e}")
                    else:
                        st.error(f"❌ {message}")
            else:
                st.warning("Please provide complete PostgreSQL connection parameters.")

    st.markdown("---")

    # URL Ingestion Section
    st.header("📥 Ingest Web Content")
    input_url = st.text_input(
        "Website URL",
        placeholder="https://example.com/article",
        help="Enter any public webpage URL to scrape, vectorize, and index into PostgreSQL.",
    )

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk Size", value=1000, step=100, min_value=200, max_value=4000)
    with col2:
        chunk_overlap = st.number_input("Overlap", value=200, step=50, min_value=0, max_value=1000)

    if st.button("🚀 Scrape & Store Embeddings", use_container_width=True):
        if not input_url:
            st.error("Please enter a valid website URL.")
        elif not gemini_api_key:
            st.error("Please provide your Google Gemini API Key.")
        elif not db_instance:
            st.error("Please configure PostgreSQL connection settings.")
        else:
            try:
                with st.status("Processing website...", expanded=True) as status:
                    st.write("🌐 Fetching and parsing webpage...")
                    title, text = extract_text_from_url(input_url)
                    st.write(f"📄 Extracted title: **{title}** ({len(text)} characters)")

                    st.write("✂️ Chunking text...")
                    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.write(f"📦 Created **{len(chunks)}** chunks")

                    if not chunks:
                        status.update(label="No text content found at URL!", state="error")
                    else:
                        st.write("🧠 Generating Gemini embeddings...")
                        gemini_svc = GeminiService(api_key=gemini_api_key)
                        embeddings = gemini_svc.get_embeddings_batch(chunks, model=embedding_model)

                        st.write("💾 Storing vectors into PostgreSQL...")
                        db_instance.init_db(dimension=768)
                        stored_count = db_instance.store_url_chunks(
                            url=input_url,
                            title=title,
                            chunks=chunks,
                            embeddings=embeddings,
                        )
                        st.session_state.db_connected = True
                        status.update(label=f"✅ Successfully indexed {stored_count} chunks!", state="complete")
                        st.toast(f"Successfully stored embeddings for '{title}'!", icon="🎉")
            except Exception as e:
                st.error(f"Error during ingestion: {str(e)}")

    st.markdown("---")

    # Document Upload Section
    st.header("📄 Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload document(s)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, or TXT files to extract text, vectorize, and index into PostgreSQL.",
    )

    if uploaded_files:
        col1, col2 = st.columns(2)
        with col1:
            doc_chunk_size = st.number_input("Chunk Size", value=1000, step=100, min_value=200, max_value=4000, key="doc_chunk_size")
        with col2:
            doc_chunk_overlap = st.number_input("Overlap", value=200, step=50, min_value=0, max_value=1000, key="doc_chunk_overlap")

        if st.button("🚀 Process & Store Document Embeddings", use_container_width=True):
            if not gemini_api_key:
                st.error("Please provide your Google Gemini API Key.")
            elif not db_instance:
                st.error("Please configure PostgreSQL connection settings.")
            else:
                with st.status("Processing documents...", expanded=True) as status:
                    total_stored = 0
                    for uploaded_file in uploaded_files:
                        try:
                            st.write(f"📥 Processing: **{uploaded_file.name}**")
                            
                            # Save uploaded file to temporary location
                            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                                tmp_file.write(uploaded_file.getbuffer())
                                tmp_path = tmp_file.name
                            
                            try:
                                # Extract text from document
                                st.write(f"📖 Extracting text from {uploaded_file.name}...")
                                filename, text = extract_text_from_document(tmp_path)
                                st.write(f"✅ Extracted {len(text)} characters")

                                # Chunk the text
                                st.write("✂️ Chunking document...")
                                chunks = chunk_document_text(text, chunk_size=doc_chunk_size, chunk_overlap=doc_chunk_overlap)
                                st.write(f"📦 Created **{len(chunks)}** chunks")

                                if not chunks:
                                    st.warning(f"No text content found in {filename}")
                                else:
                                    # Generate embeddings
                                    st.write("🧠 Generating Gemini embeddings...")
                                    gemini_svc = GeminiService(api_key=gemini_api_key)
                                    embeddings = gemini_svc.get_embeddings_batch(chunks, model=embedding_model)

                                    # Store in database
                                    st.write(f"💾 Storing vectors into PostgreSQL...")
                                    db_instance.init_db(dimension=768)
                                    stored_count = db_instance.store_document_chunks(
                                        document_id=uploaded_file.name,
                                        title=filename,
                                        chunks=chunks,
                                        embeddings=embeddings,
                                    )
                                    total_stored += stored_count
                                    st.write(f"✅ Successfully indexed {stored_count} chunks from {filename}")
                            finally:
                                # Clean up temporary file
                                if os.path.exists(tmp_path):
                                    os.remove(tmp_path)
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    
                    if total_stored > 0:
                        st.session_state.db_connected = True
                        status.update(label=f"✅ Successfully indexed {total_stored} chunks from all documents!", state="complete")
                        st.toast(f"Successfully stored embeddings for {len(uploaded_files)} document(s)!", icon="🎉")

    st.markdown("---")

    # Stored URLs / Knowledge Base manager
    st.header("📚 Knowledge Base")
    indexed_urls = []
    if db_instance:
        try:
            indexed_urls = db_instance.list_indexed_urls()
        except Exception:
            indexed_urls = []

    if indexed_urls:
        st.write(f"**Total Indexed Sources:** {len(indexed_urls)}")
        for item in indexed_urls:
            with st.container():
                st.markdown(f"**[{item['title']}]({item['url']})**")
                st.caption(f"Chunks: {item['chunk_count']} | Last updated: {item['last_indexed']}")
                if st.button("🗑️ Remove", key=f"del_{item['url']}"):
                    db_instance.delete_url(item["url"])
                    st.rerun()

        if st.button("⚠️ Clear Entire Database", type="secondary", use_container_width=True):
            db_instance.clear_all()
            st.success("All stored embeddings have been cleared.")
            st.rerun()
    else:
        st.info("No URLs indexed yet. Ingest a URL above to start asking questions!")


# ----------------- MAIN CHAT VIEW -----------------
st.markdown('<div class="main-title">🌐 Web URL Knowledge Bot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Scrape web content, store vector embeddings in PostgreSQL (pgvector), and chat with Google Gemini.</div>',
    unsafe_allow_html=True,
)

# Top Bar Filters & Options
filter_col1, filter_col2, filter_col3 = st.columns([3, 2, 1])

with filter_col1:
    url_options = ["All Indexed URLs"] + [item["url"] for item in indexed_urls]
    selected_source = st.selectbox("🎯 Target Knowledge Source:", options=url_options, index=0)

with filter_col2:
    top_k = st.slider("🔍 Context chunks to retrieve (Top-K):", min_value=1, max_value=10, value=4)

with filter_col3:
    st.write("")
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔎 View Retrieved Context Sources"):
                for idx, src in enumerate(message["sources"], start=1):
                    st.markdown(f"**Source [{idx}]:** [{src.get('title') or src.get('url')}]({src.get('url')}) (Similarity: `{src.get('similarity', 0):.4f}`)")
                    st.text(src.get("chunk_text", "")[:300] + "..." if len(src.get("chunk_text", "")) > 300 else src.get("chunk_text", ""))

# Chat Input Handler
if prompt := st.chat_input("Ask a question about your indexed websites..."):
    # Check prerequisites
    if not gemini_api_key:
        st.warning("⚠️ Please provide a Google Gemini API Key in the sidebar.")
    elif not db_instance:
        st.warning("⚠️ Please configure PostgreSQL connection in the sidebar.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Searching PostgreSQL vectors and generating answer with Gemini..."):
                try:
                    gemini_svc = GeminiService(api_key=gemini_api_key)

                    # 1. Embed query
                    query_embedding = gemini_svc.get_query_embedding(prompt, model=embedding_model)

                    # 2. Search PostgreSQL pgvector
                    url_filter = None if selected_source == "All Indexed URLs" else selected_source
                    context_chunks = db_instance.search_similar(
                        query_embedding=query_embedding,
                        top_k=top_k,
                        url_filter=url_filter,
                    )

                    # 3. Generate response with Gemini
                    response_text = gemini_svc.generate_rag_response(
                        query=prompt,
                        context_chunks=context_chunks,
                        model_name=gemini_chat_model,
                    )

                    # Display result
                    st.markdown(response_text)

                    # Display sources expander
                    if context_chunks:
                        with st.expander("🔎 View Retrieved Context Sources"):
                            for idx, src in enumerate(context_chunks, start=1):
                                st.markdown(f"**Source [{idx}]:** [{src.get('title') or src.get('url')}]({src.get('url')}) (Similarity: `{src.get('similarity', 0):.4f}`)")
                                st.text(src.get("chunk_text", "")[:300] + "..." if len(src.get("chunk_text", "")) > 300 else src.get("chunk_text", ""))

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": context_chunks,
                    })

                except Exception as e:
                    error_msg = f"❌ An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
