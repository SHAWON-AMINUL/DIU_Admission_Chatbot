"""Wire shapes for admission documents.

A "document" here is a row in `posts` -- the table name is the Backend
teammate's and is not ours to rename, but everything the AI side does with it
is document-shaped: it gets chunked, embedded and retrieved.

These models are the contract. Declaring them as FastAPI `response_model`s is
what removes the hand-built dicts the endpoints used to return, where a field
was addressed as `r[11]` and inserting a column silently shifted every value
after it into the wrong key.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class IngestStats(BaseModel):
    """How far a document has got through the pipeline.

    chunk_count and vector_count are reported separately on purpose: equal
    counts mean the document is searchable, while chunks without vectors mean
    it is indexed but invisible to vector search -- a state that otherwise
    looks healthy from the outside.
    """

    chunk_count: int = 0
    vector_count: int = 0
    token_total: int = 0


class PostSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    body: str | None = None
    status: str
    approval_status: str
    verify_status: str | None = None
    fail_count: int = 0
    created_at: datetime | None = None
    published_at: datetime | None = None
    approved_at: datetime | None = None
    verified_at: datetime | None = None
    ingest: IngestStats = IngestStats()


class PostCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    body: str = Field(default="")
    category_ids: list[int] = Field(default_factory=list)


class ChunkOut(BaseModel):
    id: int
    chunk_index: int
    chunk_text: str
    token_count: int
    model_name: str | None = None
    has_vector: bool = False


class FailureOut(BaseModel):
    reason: str | None = None
    chunk_index: int | None = None
    at: datetime
