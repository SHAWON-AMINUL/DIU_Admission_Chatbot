"""Editorial workflow and ingest triggers for admission documents.

Each endpoint now does one thing: translate HTTP to a repository call and a
status code. No SQL, no connection handling, no tuple unpacking.

In the real system the Backend teammate owns the editorial endpoints; the
piece that belongs to this slice is the ingest trigger. It is fire-and-forget
so a document with dozens of chunks never holds the verify request open, and
it is the seam their verify handler will call.
"""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.api.dependencies import get_post_repository
from app.database.repositories import PostRepository
from app.ingestion.pipeline import ingest_post
from app.schemas.document import ChunkOut, FailureOut, PostCreate, PostSummary

router = APIRouter(prefix="/api", tags=["documents"])

Repo = Annotated[PostRepository, Depends(get_post_repository)]


@router.get("/posts", response_model=list[PostSummary])
def list_posts(repo: Repo) -> list[PostSummary]:
    return repo.list_summaries()


@router.post("/posts")
def create_post(payload: PostCreate, repo: Repo) -> dict:
    """Step 1-2: create a Draft document (+ category links)."""
    post_id = repo.create(payload.title, payload.body, payload.category_ids)
    if post_id is None:
        raise HTTPException(500, "no users exist; run the seed script first")
    return {"id": post_id, "status": "Draft"}


@router.post("/posts/{post_id}/publish")
def publish(post_id: int, repo: Repo) -> dict:
    """Step 3: Draft -> published."""
    if not repo.publish(post_id):
        raise HTTPException(404, "post not found")
    return {"id": post_id, "status": "published"}


@router.post("/posts/{post_id}/approve")
def approve(post_id: int, repo: Repo) -> dict:
    """Director-level approval gate. Must clear before verification."""
    if not repo.approve(post_id):
        raise HTTPException(404, "post not found")
    return {"id": post_id, "approval_status": "approved"}


@router.post("/posts/{post_id}/verify")
def verify(post_id: int, background: BackgroundTasks, repo: Repo) -> dict:
    """Step 4-5: verify, then queue ingest."""
    status = repo.approval_status(post_id)
    if status is None:
        raise HTTPException(404, "post not found")
    if status != "approved":
        raise HTTPException(
            409, "post must be approved by a director before verification"
        )

    repo.mark_verified(post_id)
    repo.commit()
    background.add_task(ingest_post, post_id)
    return {"id": post_id, "verify_status": "verified", "ingest": "queued"}


@router.post("/posts/{post_id}/ingest")
def reingest(post_id: int, background: BackgroundTasks) -> dict:
    """Re-run ingest by hand -- also how a failed document is retried."""
    background.add_task(ingest_post, post_id)
    return {"id": post_id, "ingest": "queued"}


@router.get("/posts/{post_id}/chunks", response_model=list[ChunkOut])
def post_chunks(post_id: int, repo: Repo) -> list[ChunkOut]:
    return repo.chunks(post_id)


@router.get("/posts/{post_id}/failures", response_model=list[FailureOut])
def post_failures(post_id: int, repo: Repo) -> list[FailureOut]:
    return repo.failures(post_id)
