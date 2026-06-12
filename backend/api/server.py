"""
FastAPI Server
==============
REST API for the Video Note Extractor.

Endpoints:
  POST /extract         — submit a video source, get full extraction
  POST /extract/async   — submit a job, poll with GET /jobs/{id}
  GET  /jobs/{job_id}   — check job status + get result when ready
  GET  /health          — health check

Run with:
  uvicorn backend.api.server:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
"""

import asyncio
import uuid
import time
import logging
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.core.pipeline import VideoNotePipeline

logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────

app = FastAPI(
    title="Video Note Extractor API",
    description="Transforms long videos into organised notes, timestamps, and action items.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (use Redis in production)
JOBS: dict = {}

# ── Request / Response schemas ────────────────────

class ExtractRequest(BaseModel):
    source: str = Field(..., description="YouTube URL or local file path")
    whisper_model: str = Field("base", description="tiny|base|small|medium|large")
    llm_provider: str  = Field("openai", description="openai|anthropic")
    llm_model: str     = Field("gpt-4o")
    openai_api_key: Optional[str]     = None
    anthropic_api_key: Optional[str]  = None
    language: Optional[str]           = None

class JobStatus(BaseModel):
    job_id: str
    status: str       # "pending" | "processing" | "done" | "error"
    progress: int     # 0-100
    message: str
    result: Optional[dict] = None
    error: Optional[str]   = None

# ── Endpoints ─────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/extract", response_model=dict, summary="Synchronous extraction (blocks until done)")
async def extract_sync(req: ExtractRequest):
    """
    Synchronous extraction — waits for the full pipeline to complete.
    Suitable for short videos (<5 min). For longer videos, use /extract/async.
    """
    try:
        pipeline = _build_pipeline(req)
        result = await asyncio.get_event_loop().run_in_executor(
            None, pipeline.run, req.source
        )
        return result.to_dict()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/async", response_model=JobStatus,
          summary="Asynchronous extraction — returns job ID immediately")
async def extract_async(req: ExtractRequest, background_tasks: BackgroundTasks):
    """
    Submit an extraction job.  Returns immediately with a job_id.
    Poll GET /jobs/{job_id} to check progress and retrieve results.
    """
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "pending", "progress": 0,
        "message": "Job queued", "result": None, "error": None
    }
    background_tasks.add_task(_run_job, job_id, req)
    return JobStatus(job_id=job_id, status="pending", progress=0, message="Job queued")


@app.get("/jobs/{job_id}", response_model=JobStatus,
         summary="Poll job status")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    j = JOBS[job_id]
    return JobStatus(job_id=job_id, **j)


@app.post("/extract/file", summary="Upload a video/audio file and extract")
async def extract_file(
    file: UploadFile = File(...),
    whisper_model: str = Form("base"),
    llm_model: str = Form("gpt-4o"),
    openai_api_key: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Upload a video/audio file.  Saves it temporarily and runs extraction.
    """
    tmp_dir = Path("/tmp/vne_uploads")
    tmp_dir.mkdir(exist_ok=True)
    save_path = tmp_dir / file.filename

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    req = ExtractRequest(
        source=str(save_path),
        whisper_model=whisper_model,
        llm_model=llm_model,
        openai_api_key=openai_api_key,
    )
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "pending", "progress": 0,
        "message": "File uploaded, job queued", "result": None, "error": None
    }
    background_tasks.add_task(_run_job, job_id, req)
    return {"job_id": job_id, "filename": file.filename, "status": "pending"}


# ── Background task ───────────────────────────────

async def _run_job(job_id: str, req: ExtractRequest):
    """Background coroutine that runs the pipeline and updates JOBS[job_id]."""
    JOBS[job_id].update({"status": "processing", "progress": 10,
                          "message": "Extracting audio …"})
    try:
        pipeline = _build_pipeline(req)

        # Step progress updates (since pipeline.run is synchronous, we update before/after)
        loop = asyncio.get_event_loop()

        JOBS[job_id].update({"progress": 25, "message": "Transcribing with Whisper …"})
        result = await loop.run_in_executor(None, pipeline.run, req.source)

        JOBS[job_id].update({
            "status":   "done",
            "progress": 100,
            "message":  "Extraction complete",
            "result":   result.to_dict(),
        })
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        JOBS[job_id].update({"status": "error", "progress": 0,
                              "message": "Failed", "error": str(e)})


def _build_pipeline(req: ExtractRequest) -> VideoNotePipeline:
    return VideoNotePipeline(
        openai_api_key=req.openai_api_key,
        anthropic_api_key=req.anthropic_api_key,
        whisper_model=req.whisper_model,
        llm_provider=req.llm_provider,
        llm_model=req.llm_model,
        language=req.language,
        verbose=False,
    )