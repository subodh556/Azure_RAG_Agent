"""
Azure AI Search index lifecycle management.

Creates the index schema:
  • BM25 full-text search  (en.microsoft analyser)
  • HNSW vector search     (cosine, m=4, efConstruction=400)
  • Semantic reranker       (cross-encoder on top of RRF)
"""
from __future__ import annotations

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from config import AzureConfig


class AzureSearchIndexManager:
    """
    Manages the Azure AI Search index via SearchIndexClient.

    Index schema
    ────────────
      chunk_id      String  key, filterable
      doc_id        String  filterable, facetable
      doc_name      String  searchable, filterable, facetable
      chunk_index   Int32   filterable, sortable
      page_num      Int32   filterable, sortable
      text          String  searchable (en.microsoft analyzer, BM25)
      text_vector   Collection(Single)  1536-dim HNSW cosine vector field
    """

    ALGO_NAME    = "rag-hnsw-algo"
    PROFILE_NAME = "rag-vector-profile"

    def __init__(self, cfg: AzureConfig) -> None:
        self.cfg     = cfg
        self._client = SearchIndexClient(
            endpoint   = cfg.search_endpoint,
            credential = AzureKeyCredential(cfg.search_api_key),
        )

    def _schema(self) -> SearchIndex:
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name       = self.ALGO_NAME,
                    parameters = HnswParameters(
                        m               = 4,
                        ef_construction = 400,
                        ef_search       = 500,
                        metric          = "cosine",
                    ),
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name                         = self.PROFILE_NAME,
                    algorithm_configuration_name = self.ALGO_NAME,
                )
            ],
        )

        semantic_search = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name               = self.cfg.semantic_config,
                    prioritized_fields = SemanticPrioritizedFields(
                        content_fields = [SemanticField(field_name="text")],
                        keywords_fields= [SemanticField(field_name="doc_name")],
                    ),
                )
            ]
        )

        fields = [
            SimpleField(
                name="chunk_id", type=SearchFieldDataType.String,
                key=True, filterable=True,
            ),
            SimpleField(
                name="doc_id", type=SearchFieldDataType.String,
                filterable=True, facetable=True,
            ),
            SearchableField(
                name="doc_name", type=SearchFieldDataType.String,
                filterable=True, facetable=True,
            ),
            SimpleField(
                name="chunk_index", type=SearchFieldDataType.Int32,
                filterable=True, sortable=True,
            ),
            SimpleField(
                name="page_num", type=SearchFieldDataType.Int32,
                filterable=True, sortable=True,
            ),
            SearchableField(
                name="text", type=SearchFieldDataType.String,
                analyzer_name="en.microsoft",
            ),
            SearchField(
                name="text_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions   = self.cfg.embed_dimensions,
                vector_search_profile_name = self.PROFILE_NAME,
            ),
        ]

        return SearchIndex(
            name            = self.cfg.index_name,
            fields          = fields,
            vector_search   = vector_search,
            semantic_search = semantic_search,
        )

    def create_or_update_index(self) -> None:
        self._client.create_or_update_index(self._schema())

    def delete_index(self) -> None:
        try:
            self._client.delete_index(self.cfg.index_name)
        except ResourceNotFoundError:
            pass

    def index_exists(self) -> bool:
        try:
            self._client.get_index(self.cfg.index_name)
            return True
        except ResourceNotFoundError:
            return False

    def get_doc_count(self) -> int:
        try:
            return self._client.get_index_statistics(self.cfg.index_name).document_count
        except Exception:
            return -1
