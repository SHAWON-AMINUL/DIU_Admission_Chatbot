"""Client for the embedding microservice.

The ingest pipeline and the query side both go through this, so neither ever
loads model weights of its own.
"""

import httpx

from app.config import settings


class EmbeddingClient:
    # Generous: a large post's batch takes minutes if the service falls back
    # to CPU (EMBEDDING_DEVICE=cpu, or a machine without MPS). The service
    # preloads its model, so this no longer has to cover a cold start as well.
    def __init__(self, base_url: str | None = None, timeout: float = 300.0):
        self.base_url = (base_url or settings.EMBEDDING_SERVICE_URL).rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def health(self) -> dict:
        r = self._client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def embed(self, texts: list[str]) -> tuple[str, list[list[float]]]:
        """Return (model_name_tag, vectors). Vectors are already L2-normalized."""
        r = self._client.post(f"{self.base_url}/embed", json={"texts": texts})
        r.raise_for_status()
        payload = r.json()

        if payload["dim"] != settings.EMBEDDING_DIM:
            raise ValueError(
                f"Embedding service returned dim={payload['dim']}, but the "
                f"ai_content_vectors.vector column is vector({settings.EMBEDDING_DIM}). "
                "Inserting would fail; refusing to continue."
            )
        return payload["model_name"], payload["embeddings"]

    def token_counts(self, texts: list[str]) -> list[int]:
        r = self._client.post(f"{self.base_url}/token_count", json={"texts": texts})
        r.raise_for_status()
        return r.json()["counts"]

    def close(self) -> None:
        self._client.close()
