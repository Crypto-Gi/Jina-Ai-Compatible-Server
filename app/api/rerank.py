"""
Jina Local API Server - Rerank API Endpoint
===========================================
POST /v1/rerank - Rerank documents compatible with Jina API.
"""

from fastapi import APIRouter, HTTPException, Request

from app.logging_config import get_logger
from app.schemas.embeddings import ErrorResponse
from app.schemas.rerank import (
    DocumentObject,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
)

logger = get_logger(__name__)

router = APIRouter(tags=["Rerank"])


@router.post(
    "/rerank",
    response_model=RerankResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
        422: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def rerank_documents(request: Request, payload: RerankRequest) -> RerankResponse:
    """
    Rerank documents by relevance to a query.
    
    Compatible with the Jina AI Reranker API.
    
    Documents are returned sorted by relevance_score (highest first).
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.info(
        "Rerank request received",
        request_id=request_id,
        model=payload.model,
        query_length=len(payload.query),
        document_count=len(payload.documents),
        top_n=payload.top_n,
    )

    # Get model from registry
    registry = request.app.state.registry
    model = registry.get_reranker_model(payload.model)
    
    if model is None:
        logger.warning("Model not found", model=payload.model)
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{payload.model}' not found or not loaded",
                    "type": "invalid_request_error",
                    "param": "model",
                }
            },
        )

    # Validate input
    if not payload.query.strip():
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "message": "Query cannot be empty",
                    "type": "invalid_request_error",
                    "param": "query",
                }
            },
        )

    if not payload.documents:
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "message": "Documents list cannot be empty",
                    "type": "invalid_request_error",
                    "param": "documents",
                }
            },
        )

    try:
        # Rerank documents
        raw_results, token_count = model.rerank(
            query=payload.query,
            documents=payload.documents,
            top_n=payload.top_n,
            return_documents=payload.return_documents,
        )

        # Format results
        results = []
        for r in raw_results:
            result = RerankResult(
                index=r["index"],
                relevance_score=r["relevance_score"],
                document=DocumentObject(text=r["document"]["text"]) if r.get("document") else None,
            )
            results.append(result)

        response = RerankResponse(
            model=payload.model,
            results=results,
            usage=RerankUsage(total_tokens=token_count),
        )

        logger.info(
            "Rerank completed",
            request_id=request_id,
            model=payload.model,
            result_count=len(results),
            top_score=results[0].relevance_score if results else 0,
            tokens=token_count,
        )

        return response

    except Exception as e:
        logger.error(
            "Rerank failed",
            request_id=request_id,
            model=payload.model,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Reranking failed: {str(e)}",
                    "type": "internal_error",
                }
            },
        )
