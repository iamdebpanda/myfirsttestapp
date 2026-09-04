from typing import Any, Dict, List, Optional
import google.generativeai as genai


class GeminiService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)

    def get_embedding(self, text: str, model: str = "gemini-embedding-2", task_type: str = "retrieval_document") -> List[float]:
        """Generate a single vector embedding for the input text."""
        response = genai.embed_content(
            model=model,
            content=text, 
            task_type=task_type,
            config=genai.EmbedContentConfig(output_dimensionality=768),
            
        )
        return response["embedding"]

    def get_embeddings_batch(
        self,
        texts: List[str],
        model: str = "gemini-embedding-2",
        task_type: str = "retrieval_document",
        batch_size: int = 50,
    ) -> List[List[float]]:
        """Generate embeddings in batches for a list of text chunks."""
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = genai.embed_content(
                model=model,
                content=batch,
                task_type=task_type,
            )
            embeddings.extend(response["embedding"])
        return embeddings

    def get_query_embedding(self, query: str, model: str = "gemini-embedding-2") -> List[float]:
        """Generate vector embedding for a search query."""
        response = genai.embed_content(
            model=model,
            content=query,
            task_type="retrieval_query",
        )
        return response["embedding"]

    def generate_rag_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        model_name: str = "gemini-3.8-flash",
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Synthesize an answer using Google Gemini based on retrieved context chunks and chat history.
        """
        # Format context from retrieved chunks
        context_text_blocks = []
        for idx, chunk in enumerate(context_chunks, start=1):
            url = chunk.get("url", "Unknown URL")
            title = chunk.get("title", "")
            text = chunk.get("chunk_text", "")
            context_text_blocks.append(
                f"--- Document Source [{idx}] ---\nTitle: {title}\nURL: {url}\nContent:\n{text}\n"
            )

        context_str = "\n".join(context_text_blocks) if context_text_blocks else "No relevant context found in the database."

        default_instruction = (
            "You are a helpful and accurate AI assistant powered by Google Gemini and PostgreSQL vector search. "
            "Your task is to answer user questions using the provided source context chunks scraped from websites. "
            "Guidelines:\n"
            "1. Ground your answer in the provided context as much as possible.\n"
            "2. If the context doesn't have sufficient information to answer the question, state that clearly and provide the best general knowledge answer while clarifying the limitation.\n"
            "3. Cite your sources by mentioning the relevant URL(s) or titles when applicable.\n"
            "4. Keep your answer well-structured with markdown headings, bullet points, and code blocks where appropriate."
        )

        instruction = system_instruction or default_instruction

        prompt = f"""
System Instructions:
{instruction}

Retrieved Web Content Context:
{context_str}

User Question:
{query}
"""

        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(prompt)
        return response.text
