"""
Jina Local API Server - Main Application
========================================
FastAPI application with lifespan management, health endpoints, and routers.
"""

# IMPORTANT: Set these env vars BEFORE importing anything else
# to disable flash attention which causes segfaults on newer GPUs (e.g., RTX 5090)
import os
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
os.environ["USE_FLASH_ATTN"] = "FALSE"

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import embeddings, models, rerank
from app.config import settings
from app.logging_config import (
    bind_request_context,
    clear_request_context,
    get_logger,
    setup_logging,
)
from app.models.registry import ModelRegistry

# Initialize logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles:
    - Model loading on startup
    - Cleanup on shutdown
    """
    logger.info(
        "Starting Jina Local API Server",
        device=settings.device,
        models_to_load=settings.models_to_load,
    )

    # Initialize model registry and load models
    registry = ModelRegistry(settings)
    await registry.load_models()
    app.state.registry = registry

    logger.info(
        "Server startup complete",
        loaded_models=registry.loaded_models(),
    )

    yield

    # Shutdown
    logger.info("Shutting down server...")
    registry.unload_all()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Jina Local API Server",
    description="Local server providing Jina AI compatible embeddings and reranking APIs",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware (permissive for local use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request context middleware
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """
    Add request ID and timing to all requests.
    
    Features:
    - Generates unique request ID
    - Binds request ID to logging context
    - Measures request duration
    - Adds X-Request-ID header to response
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    bind_request_context(request_id)

    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Request completed",
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "Request failed",
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            error=str(e),
            duration_ms=round(duration_ms, 2),
        )
        raise

    finally:
        clear_request_context()


# ===========================================
# Health Endpoints
# ===========================================


@app.get("/healthz", tags=["Health"])
async def health_check():
    """
    Basic health check endpoint.
    
    Returns 200 if the server is running.
    Used by container health checks and load balancers.
    """
    return {"status": "ok"}


@app.get("/readyz", tags=["Health"])
async def readiness_check(request: Request):
    """
    Readiness check endpoint.
    
    Returns 200 if at least one model is loaded and ready.
    Returns 503 if no models are available.
    """
    registry = request.app.state.registry

    if registry.is_ready():
        return {
            "status": "ready",
            "models": registry.loaded_models(),
            "model_count": len(registry.loaded_models()),
        }

    return JSONResponse(
        status_code=503,
        content={
            "status": "not_ready",
            "message": "No models loaded",
        },
    )


@app.get("/", tags=["Info"])
async def root():
    """
    Root endpoint with server information.
    """
    return {
        "name": "Jina Local API Server",
        "version": "1.0.0",
        "description": "Local Jina-compatible embeddings and reranking API",
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "rerank": "/v1/rerank",
            "models": "/v1/models",
            "health": "/healthz",
            "readiness": "/readyz",
            "docs": "/docs",
        },
    }


# ===========================================
# Include API Routers
# ===========================================

app.include_router(embeddings.router, prefix="/v1")
app.include_router(rerank.router, prefix="/v1")
app.include_router(models.router, prefix="/v1")


# ===========================================
# Error Handlers
# ===========================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for uncaught errors.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "Unhandled exception",
        request_id=request_id,
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "request_id": request_id,
            }
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        workers=1,
    )
