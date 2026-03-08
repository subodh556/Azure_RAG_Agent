"""
Azure service credentials and tuning knobs.
All values are loaded from environment variables / .env file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # real env vars always win over .env
except ImportError:
    pass
    
@dataclass
class AzureConfig:
    """All service credentials and tuning knobs, loaded from environment."""
    search_endpoint:   str
    search_api_key:    str
    index_name:        str
    openai_endpoint:   str
    openai_api_key:    str
    embed_deployment:  str
    embed_dimensions:  int
    chat_deployment:   str        # GPT-4o mini deployment name
    top_k:             int
    use_semantic_rank: bool
    semantic_config:   str = "rag-semantic-config"

    @classmethod
    def from_env(cls) -> "AzureConfig":
        missing: list[str] = []

        def req(k: str) -> str:
            v = os.environ.get(k, "").strip()
            if not v:
                missing.append(k)
            return v

        cfg = cls(
            search_endpoint   = req("AZURE_SEARCH_ENDPOINT"),
            search_api_key    = req("AZURE_SEARCH_API_KEY"),
            index_name        = os.environ.get("RAG_INDEX_NAME", "rag-documents"),
            openai_endpoint   = req("AZURE_OPENAI_ENDPOINT"),
            openai_api_key    = req("AZURE_OPENAI_API_KEY"),
            embed_deployment  = os.environ.get("AZURE_OPENAI_EMBED_DEP", "text-embedding-3-small"),
            embed_dimensions  = int(os.environ.get("EMBED_DIMENSIONS", "1536")),
            chat_deployment   = os.environ.get("AZURE_OPENAI_CHAT_DEP", "gpt-4o-mini"),
            top_k             = int(os.environ.get("TOP_K", "5")),
            use_semantic_rank = os.environ.get("USE_SEMANTIC_RANK", "true").lower() == "true",
        )
        if missing:
            raise EnvironmentError(
                f"Missing environment variables: {', '.join(missing)}\n"
            )
        return cfg
