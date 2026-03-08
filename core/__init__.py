"""core — Azure service wrappers: index management, embeddings, chunking, retrieval, PDF extraction."""
from .index_manager import AzureSearchIndexManager
from .embeddings    import EmbeddingService
from .chunker       import DocumentChunker
from .retriever     import AzureSearchRetriever
from .pdf_extractor import PDFExtractor

__all__ = [
    "AzureSearchIndexManager",
    "EmbeddingService",
    "DocumentChunker",
    "AzureSearchRetriever",
    "PDFExtractor",
]
