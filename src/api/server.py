"""FastAPI server for the AI-Driven VR Learning System.

Provides REST API endpoints and WebSocket support for real-time
interaction between the smart glasses frontend and the AI pipeline.
"""

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..pipeline import LearningPipeline
from ..utils.config import load_config

logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: LearningPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup."""
    global pipeline
    config = load_config()
    pipeline = LearningPipeline(config)
    pipeline.initialize()
    logger.info("Learning pipeline ready")
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI-Driven VR Learning System",
        description="API for the AI-powered smart educational glasses platform",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files for frontend
    import os
    from pathlib import Path
    static_dir = Path(__file__).parent.parent.parent / "frontend" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Register routes
    _register_routes(app)

    return app


# --- Request/Response Models ---

class QueryRequest(BaseModel):
    text: str
    learner_id: str = "default"

class FeedbackRequest(BaseModel):
    learner_id: str = "default"
    comprehension_change: float = 0.0
    engagement: float = 0.5
    interaction_time: float = 0.0
    quiz_score: float | None = None
    domain: str | None = None
    follow_up: bool = False
    verbal_feedback: str | None = None
    rating: int | None = None

class QuizAnswerRequest(BaseModel):
    learner_id: str = "default"
    domain: str
    question_index: int
    answer_index: int


def _register_routes(app: FastAPI):
    """Register all API routes."""

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve the main frontend page."""
        from pathlib import Path
        template_path = Path(__file__).parent.parent.parent / "frontend" / "templates" / "index.html"
        if template_path.exists():
            return template_path.read_text()
        return HTMLResponse("<h1>AI-Driven VR Learning System</h1><p>Frontend not built yet.</p>")

    @app.get("/api/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "pipeline_ready": pipeline is not None and pipeline._initialized,
            "timestamp": time.time(),
        }

    @app.post("/api/query")
    async def process_query(request: QueryRequest):
        """Process a text query through the learning pipeline.

        This is the primary endpoint for the smart glasses:
        text query -> semantic analysis -> knowledge reasoning ->
        VR visualization -> response
        """
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        response = pipeline.process_text(request.text, request.learner_id)
        return response.to_dict()

    @app.post("/api/feedback")
    async def submit_feedback(request: FeedbackRequest):
        """Submit learner feedback for adaptive learning."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        feedback_data = request.model_dump(exclude_none=True)
        result = pipeline.process_feedback(request.learner_id, feedback_data)
        return result

    @app.get("/api/learner/{learner_id}")
    async def get_learner(learner_id: str):
        """Get learner profile and statistics."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        return pipeline.get_learner_stats(learner_id)

    @app.post("/api/session/reset")
    async def reset_session(learner_id: str = "default"):
        """Reset the learning session for a new conversation."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        pipeline.reset_session(learner_id)
        return {"status": "session_reset", "learner_id": learner_id}

    @app.get("/api/domains")
    async def list_domains():
        """List available educational domains and their concepts."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        domains = {}
        ontology = pipeline.knowledge_engine.ontology
        for domain in ["chemistry", "biology", "physics", "mathematics", "anatomy", "engineering"]:
            concepts = ontology.get_by_domain(domain)
            domains[domain] = [
                {"id": c.id, "name": c.name, "difficulty": c.difficulty_level}
                for c in concepts
            ]
        return {"domains": domains}

    @app.websocket("/ws/{learner_id}")
    async def websocket_endpoint(websocket: WebSocket, learner_id: str):
        """WebSocket endpoint for real-time audio streaming and interaction.

        Supports two message types:
        - text: JSON with {"type": "text", "query": "..."}
        - binary: Raw audio bytes for ASR processing
        """
        await websocket.accept()
        logger.info(f"WebSocket connected: {learner_id}")

        if pipeline:
            pipeline.reset_session(learner_id)

        try:
            while True:
                message = await websocket.receive()

                if "text" in message:
                    import json
                    data = json.loads(message["text"])

                    if data.get("type") == "text":
                        response = pipeline.process_text(
                            data["query"], learner_id
                        )
                        await websocket.send_json(response.to_dict())

                    elif data.get("type") == "feedback":
                        result = pipeline.process_feedback(learner_id, data)
                        await websocket.send_json(result)

                    elif data.get("type") == "reset":
                        pipeline.reset_session(learner_id)
                        await websocket.send_json({"status": "session_reset"})

                elif "bytes" in message:
                    # Process raw audio bytes
                    audio_bytes = message["bytes"]
                    response = pipeline.process_audio(
                        audio_bytes, learner_id
                    )
                    await websocket.send_json(response.to_dict())

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {learner_id}")
