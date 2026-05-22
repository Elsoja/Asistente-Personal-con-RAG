from __future__ import annotations

import os
import re
import glob as globmod
from typing import Any
import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Default configs
DEFAULT_DATA_DIR = "data"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama3.2"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32
DEFAULT_TOP_K = 4
OVERFETCH_MULTIPLIER = 3


TAG_TO_TYPE = {
    "/email": "emails",
    "/emails": "emails",
    "/notes": "notes",
    "/note": "notes",
    "/sms": "sms",
    "/calendar": "calendar",
}


def _parse_int_setting(name: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    return parsed


def resolve_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolves runtime configuration with defaults and typed settings."""
    config = config or {}

    resolved = {
        "api_key": config.get("api_key", None),
        "base_url": config.get("base_url", None),
        "model": config.get("model", DEFAULT_LLM_MODEL),
        "embedding_model": config.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "top_k": _parse_int_setting(
            "TOP_K",
            config.get("top_k", DEFAULT_TOP_K),
        ),
        "chunk_size": _parse_int_setting(
            "CHUNK_SIZE",
            config.get("chunk_size", DEFAULT_CHUNK_SIZE),
        ),
        "chunk_overlap": _parse_int_setting(
            "CHUNK_OVERLAP",
            config.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
        ),
    }

    if resolved["top_k"] <= 0:
        raise ValueError("TOP_K must be > 0")
    if resolved["chunk_size"] <= 0:
        raise ValueError("CHUNK_SIZE must be > 0")
    if resolved["chunk_overlap"] < 0:
        raise ValueError("CHUNK_OVERLAP must be >= 0")
    if resolved["chunk_overlap"] >= resolved["chunk_size"]:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    return resolved


def load_documents(data_dir: str = DEFAULT_DATA_DIR) -> list[Document]:
    """Loads documents from the personal data folders.

    The collection contains one LangChain Document per `.txt` file in the
    emails, notes, SMS, and calendar folders. Each document stores the file text
    as `page_content` and includes metadata for the source file path and
    document type.
    """
    documents: list[Document] = []
    folder_types = ["emails", "notes", "sms", "calendar"]

    for doc_type in folder_types:
        folder_path = os.path.join(data_dir, doc_type)
        pattern = os.path.join(folder_path, "*.txt")

        for file_path in sorted(globmod.glob(pattern)):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "type": doc_type,
                    },
                )
            )

    return documents


def split_documents(
        docs: list[Document],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Splits documents into overlapping chunks.

    The resulting chunked Document objects use the configured chunk size and
    overlap while preserving the original document metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def build_index(
        chunks: list[Document],
        embedding_model: SentenceTransformer,
) -> faiss.IndexFlatIP:
    """Creates a FAISS inner-product index for embedded document chunks.

    The index contains normalized float32 embeddings generated from each
    chunk's text with the provided embedding model.
    """
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def parse_type_filters(question: str) -> tuple[str, list[str]]:

    found_types: list[str] = []
    clean_question = question

    for tag, doc_type in TAG_TO_TYPE.items():
        pattern = re.compile(re.escape(tag), re.IGNORECASE)
        if pattern.search(clean_question):
            if doc_type not in found_types:
                found_types.append(doc_type)
            clean_question = pattern.sub("", clean_question)

    clean_question = re.sub(r"\s+", " ", clean_question).strip()
    return clean_question, found_types


def expand_query(question: str, client: OpenAI, model: str) -> list[str]:

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": (
                "Generate 3 alternative phrasings of this question for semantic search. "
                "Return only the questions, one per line.\n\n"
                f"Question: {question}"
            ),
        }],
    )
    raw = response.choices[0].message.content.strip()
    variants = [line.strip() for line in raw.split("\n") if line.strip()]
    return [question] + variants


def retrieve(
        query: str,
        index: faiss.IndexFlatIP,
        model: SentenceTransformer,
        chunks: list[Document],
        k: int = DEFAULT_TOP_K,
        type_filter: list[str] | None = None,
        expanded_queries: list[str] | None = None,
) -> list[dict]:
    """Gets the most relevant chunks for a query.

    Results are ordered by similarity and include the chunk text, similarity
    score, and metadata for each matching chunk.
    """
    queries = expanded_queries if expanded_queries else [query]

    fetch_k = k * OVERFETCH_MULTIPLIER if type_filter else k

    seen_indices: set[int] = set()
    all_results: list[dict] = []

    for q in queries:
        q_embedding = model.encode([q], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_embedding)

        scores, indices = index.search(q_embedding, fetch_k)

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx in seen_indices:
                continue
            seen_indices.add(idx)

            chunk_type = chunks[idx].metadata.get("type", "")
            if type_filter and chunk_type not in type_filter:
                continue

            all_results.append({
                "text": chunks[idx].page_content,
                "score": float(score),
                "metadata": chunks[idx].metadata,
            })

    all_results.sort(key=lambda r: r["score"], reverse=True)
    return all_results[:k]


SYSTEM_PROMPT = (
    "You are a personal digital assistant. You answer questions based ONLY on the "
    "context retrieved from the user's personal documents (emails, notes, SMS, and calendar). "
    "If the provided context does not contain enough information to answer the question, "
    "say that you don't have relevant information in your documents. "
    "Never fabricate information that is not in the context. "
    "Answer in the same language the user writes in."
)


class Assistant:
    """Stateful RAG assistant.

    The assistant owns the pipeline components, resolved configuration, and
    conversation history. Questions are answered with retrieved document context
    and the configured chat model.
    """

    def __init__(
            self,
            index: faiss.IndexFlatIP,
            model: SentenceTransformer,
            chunks: list[Document],
            client: OpenAI,
            config: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.model = model
        self.chunks = chunks
        self.client = client
        self.config = resolve_config(config)
        self.llm_model = self.config["model"]
        self.top_k = self.config["top_k"]
        self.history: list[dict[str, str]] = []

    def ask(self, question: str, k: int | None = None) -> str:
        """Generates an answer from the retrieved context and conversation history.

        The current question is combined with relevant document chunks, previous
        conversation messages, and the system prompt. The assistant response is
        appended to history alongside the user message.
        """
        k = k or self.top_k

        clean_question, type_filter = parse_type_filters(question)

        expanded = expand_query(clean_question, self.client, self.llm_model)

        results = retrieve(
            clean_question,
            self.index,
            self.model,
            self.chunks,
            k,
            type_filter=type_filter if type_filter else None,
            expanded_queries=expanded,
        )

        if results:
            context_parts = []
            for i, r in enumerate(results, 1):
                source = r["metadata"].get("source", "unknown")
                doc_type = r["metadata"].get("type", "unknown")
                context_parts.append(
                    f"[Document {i} | type={doc_type} | source={source}]\n{r['text']}"
                )
            context_block = "\n\n".join(context_parts)
        else:
            context_block = "No relevant documents found."

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({
            "role": "user",
            "content": f"Context:\n{context_block}\n\nQuestion: {clean_question}",
        })

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
        )
        answer = response.choices[0].message.content

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return answer

    def clear_history(self) -> None:
        """Empties the conversation history."""
        self.history.clear()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> Assistant:
        """Initializes the components required by the assistant and instantiates it

        The pipeline includes resolved configuration, loaded documents, chunked
        documents, an embedding model, a FAISS index, and an OpenAI-compatible
        client.
        """
        resolved_config = resolve_config(config)

        print("Loading documents...")
        docs = load_documents()
        print(f"  Loaded {len(docs)} documents")

        print("Splitting into chunks...")
        chunks = split_documents(
            docs,
            chunk_size=resolved_config["chunk_size"],
            chunk_overlap=resolved_config["chunk_overlap"],
        )
        print(f"  Created {len(chunks)} chunks")

        embedding_model = SentenceTransformer(resolved_config["embedding_model"])

        print("Building FAISS index...")
        index = build_index(chunks, embedding_model)
        print(f"  Indexed {index.ntotal} vectors (dim={index.d})")

        client_kwargs = {}
        if resolved_config["api_key"]:
            client_kwargs["api_key"] = resolved_config["api_key"]
        if resolved_config["base_url"]:
            client_kwargs["base_url"] = resolved_config["base_url"]
        client = OpenAI(**client_kwargs)

        print("Ready!\n")
        return cls(index, embedding_model, chunks, client, resolved_config)
