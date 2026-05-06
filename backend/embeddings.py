import logging
import uuid
from typing import List

import chromadb
from chromadb.config import Settings
from google import genai

logger = logging.getLogger(__name__)


class EmbeddingManager:
    EMBEDDING_MODEL = "models/gemini-embedding-001"

    def __init__(self, api_key: str):
        print("INIT CALLED")
        self.api_key = api_key

        self.client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1alpha"}
        )

        self.chroma_client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection_name = "media_transcripts"
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass

        return self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed_texts(self, texts: List[str]):
        embeddings = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.EMBEDDING_MODEL,
                contents=text,
                config={"task_type": "retrieval_document"}
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

    def _embed_query(self, query: str):
        response = self.client.models.embed_content(
            model=self.EMBEDDING_MODEL,
            contents=query,
            config={"task_type": "retrieval_query"}
        )
        return response.embeddings[0].values

    def add_documents(self, chunks: List[str]):
        if not chunks:
            return

        self.collection = self._get_or_create_collection()

        embeddings = self._embed_texts(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
        )

    def similarity_search(self, query: str, k: int = 4):
        try:
            query_embedding = self._embed_query(query)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count()),
                include=["documents"],
            )

            return results["documents"][0] if results["documents"] else []

        except Exception as e:
            logger.error("Similarity search failed: %s", e)
            return []

    def get_chunk_count(self):
        return self.collection.count()