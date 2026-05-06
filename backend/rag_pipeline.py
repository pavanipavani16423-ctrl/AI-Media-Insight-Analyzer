import re
import logging
from typing import List, Dict

import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from backend.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class RAGPipeline:

    def __init__(
        self,
        api_key: str,
        embedding_manager: EmbeddingManager,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        top_k: int = 4,
    ):
        self.api_key = api_key
        self.embedding_manager = embedding_manager
        self.top_k = top_k

        genai.configure(api_key=api_key)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,
            chunk_overlap=chunk_overlap * 4,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=1024,
        )

        self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        logger.info("RAGPipeline initialized (chunk_size=%d, top_k=%d)", chunk_size, top_k)

    def ingest_document(self, text: str) -> int:
        chunks = self.text_splitter.split_text(text)
        logger.info("Document split into %d chunks", len(chunks))
        self.embedding_manager.add_documents(chunks)
        logger.info("Stored %d chunks in ChromaDB", len(chunks))
        return len(chunks)

    def generate_summary(self, text: str) -> str:
        truncated = text[:8000] if len(text) > 8000 else text
        prompt = f"""You are an expert media analyst. Read the following transcript and write a clear, 
concise summary in 3-5 sentences. Cover: the main topic, key arguments or events, and the overall 
conclusion or takeaway. Be factual and objective.

TRANSCRIPT:
{truncated}

SUMMARY:"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error("Summary generation failed: %s", e)
            return f"Summary could not be generated: {str(e)}"

    def analyze_sentiment(self, text: str) -> Dict[str, str]:
        truncated = text[:6000] if len(text) > 6000 else text
        prompt = f"""Analyze the overall sentiment and emotional tone of this media transcript.

Instructions:
1. Classify the sentiment as exactly one of: Positive, Negative, Neutral, or Mixed
2. Explain WHY in 2-3 sentences
3. Respond in this EXACT format:
LABEL: <Positive|Negative|Neutral|Mixed>
EXPLANATION: <your explanation>

TRANSCRIPT:
{truncated}"""
        try:
            response = self.gemini_model.generate_content(prompt)
            raw = response.text.strip()
            label_match = re.search(r"LABEL:\s*(Positive|Negative|Neutral|Mixed)", raw, re.IGNORECASE)
            explanation_match = re.search(r"EXPLANATION:\s*(.+)", raw, re.DOTALL)
            label = label_match.group(1).capitalize() if label_match else "Neutral"
            explanation = explanation_match.group(1).strip() if explanation_match else raw
            return {"label": label, "text": explanation}
        except Exception as e:
            logger.error("Sentiment analysis failed: %s", e)
            return {"label": "Neutral", "text": f"Sentiment analysis unavailable: {str(e)}"}

    def extract_topics(self, text: str) -> List[str]:
        truncated = text[:6000] if len(text) > 6000 else text
        prompt = f"""Extract the 6-10 most important topics, themes, and key subjects from this 
media transcript. Return ONLY a comma-separated list of concise topic labels (2-4 words each).

TRANSCRIPT:
{truncated}

TOPICS:"""
        try:
            response = self.gemini_model.generate_content(prompt)
            raw = response.text.strip()
            topics = [t.strip().strip("•-*") for t in raw.split(",") if t.strip()]
            topics = [t for t in topics if 1 < len(t) < 60]
            return topics[:10]
        except Exception as e:
            logger.error("Topic extraction failed: %s", e)
            return ["Topic extraction unavailable"]

    def answer_question(self, query: str) -> str:
        retrieved_chunks = self.embedding_manager.similarity_search(query, k=self.top_k)
        if not retrieved_chunks:
            return "I couldn't find relevant information in the document to answer your question."

        context = "\n\n---\n\n".join(
            f"[Excerpt {i+1}]:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)
        )

        system_prompt = """You are a helpful AI assistant analyzing a media transcript. 
Answer questions based ONLY on the provided document excerpts. 
If the answer isn't in the excerpts, say so clearly."""

        user_prompt = f"""DOCUMENT EXCERPTS:
{context}

USER QUESTION: {query}

Please answer based on the excerpts above:"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error("Q&A generation failed: %s", e)
            return f"Error generating answer: {str(e)}"