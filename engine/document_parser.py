import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import faiss
import pymupdf4llm
from markitdown import MarkItDown
from sentence_transformers import SentenceTransformer


KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
INDEX_DIR = KNOWLEDGE_DIR / ".index"
INDEX_FILE = INDEX_DIR / "knowledge.index"
METADATA_FILE = INDEX_DIR / "knowledge_metadata.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SUPPORTED_TEXT_EXTENSIONS = {
    ".csv",
    ".md",
    ".py",
    ".txt",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
}
SUPPORTED_MARKITDOWN_EXTENSIONS = {
    ".csv",
    ".docx",
    ".xls",
    ".xlsx",
}


@dataclass
class DocumentChunk:
    source_path: str
    title: str
    chunk_id: str
    text: str


class KnowledgeBase:
    def __init__(
        self,
        knowledge_dir: Path = KNOWLEDGE_DIR,
        model_name: str = EMBEDDING_MODEL_NAME,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.index_dir = self.knowledge_dir / ".index"
        self.index_file = self.index_dir / "knowledge.index"
        self.metadata_file = self.index_dir / "knowledge_metadata.json"
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.markdown_converter = MarkItDown(enable_plugins=False)
        self.embedder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: list[DocumentChunk] = []

    def build_index(self) -> int:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        chunks = self._load_chunks_from_knowledge_folder()

        if not chunks:
            self.index = None
            self.metadata = []
            if self.index_file.exists():
                self.index_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            return 0

        embedder = self._get_embedder()
        embeddings = embedder.encode(
            [chunk.text for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, str(self.index_file))
        self.metadata_file.write_text(
            json.dumps(
                [asdict(chunk) for chunk in chunks], indent=2, ensure_ascii=True
            ),
            encoding="utf-8",
        )

        self.index = index
        self.metadata = chunks
        return len(chunks)

    def load_index(self) -> bool:
        if not self.index_file.exists() or not self.metadata_file.exists():
            return False

        self.index = faiss.read_index(str(self.index_file))
        raw_metadata = json.loads(self.metadata_file.read_text(encoding="utf-8"))
        self.metadata = [DocumentChunk(**item) for item in raw_metadata]
        return True

    def ensure_index(self) -> bool:
        return self.load_index() or self.build_index() > 0

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not isinstance(query, str):
            return []

        cleaned_query = query.strip()
        if not cleaned_query or len(cleaned_query) > 1000:
            return []

        if not self.ensure_index() or self.index is None:
            return []

        embedder = self._get_embedder()
        query_embedding = embedder.encode(
            [cleaned_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)
        results: list[dict] = []

        for score, index_position in zip(distances[0], indices[0]):
            if index_position < 0 or index_position >= len(self.metadata):
                continue

            chunk = self.metadata[index_position]
            results.append(
                {
                    "score": float(score),
                    "source_path": chunk.source_path,
                    "title": chunk.title,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                }
            )

        return results

    def search_context(self, question: str, top_k: int = 3) -> str:
        results = self.search(question, top_k=top_k)
        if not results:
            return ""

        sections = []
        for result in results:
            sections.append(
                f"Source: {result['source_path']}\n"
                f"Title: {result['title']}\n"
                f"Relevance: {result['score']:.3f}\n"
                f"{result['text']}"
            )

        return "\n\n---\n\n".join(sections)

    def _get_embedder(self) -> SentenceTransformer:
        if self.embedder is None:
            try:
                self.embedder = SentenceTransformer(
                    self.model_name, local_files_only=True
                )
            except Exception as exc:
                raise RuntimeError(
                    "The embedding model 'all-MiniLM-L6-v2' is not available in the local cache yet. "
                    "Run engine/document_parser.py once with network access to download it."
                ) from exc
        return self.embedder

    def _load_chunks_from_knowledge_folder(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        if not self.knowledge_dir.exists():
            return chunks

        for file_path in sorted(self.knowledge_dir.rglob("*")):
            if not file_path.is_file() or self.index_dir in file_path.parents:
                continue

            try:
                document_text = self._read_document(file_path)
            except Exception as exc:
                print(f"Skipping {file_path}: {exc}")
                continue

            if not document_text:
                continue

            title = file_path.stem
            for position, chunk_text in enumerate(
                self._chunk_text(document_text), start=1
            ):
                chunks.append(
                    DocumentChunk(
                        source_path=str(file_path.relative_to(self.knowledge_dir)),
                        title=title,
                        chunk_id=f"{file_path.stem}-{position}",
                        text=chunk_text,
                    )
                )

        return chunks

    def _read_document(self, file_path: Path) -> str:
        extension = file_path.suffix.lower()

        if extension == ".pdf":
            return str(pymupdf4llm.to_markdown(str(file_path))).strip()

        if extension in SUPPORTED_MARKITDOWN_EXTENSIONS:
            result = self.markdown_converter.convert_local(file_path)
            return result.markdown.strip()

        if extension in SUPPORTED_TEXT_EXTENSIONS:
            return file_path.read_text(encoding="utf-8", errors="ignore").strip()

        return ""

    def _chunk_text(self, text: str) -> list[str]:
        normalized_text = " ".join(text.split())
        if not normalized_text:
            return []

        chunks = []
        start = 0
        text_length = len(normalized_text)

        while start < text_length:
            end = min(text_length, start + self.chunk_size)
            chunk = normalized_text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = max(end - self.chunk_overlap, start + 1)

        return chunks


def main() -> None:
    knowledge_base = KnowledgeBase()
    chunk_count = knowledge_base.build_index()

    print(f"Indexed {chunk_count} chunk(s) from {knowledge_base.knowledge_dir}")

    if chunk_count == 0:
        print("Add PDFs, CSV/XLSX files, or text/code files to knowledge/ and rerun.")
        return

    sample_question = os.getenv(
        "COREBRUM_TEST_QUERY", "Summarize the available knowledge base."
    )
    print()
    print(f"Sample query: {sample_question}")
    print()
    context = knowledge_base.search_context(sample_question, top_k=3)
    print(context or "No relevant context found.")


if __name__ == "__main__":
    main()
