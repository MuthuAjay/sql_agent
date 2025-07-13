"""
FastAPI Main Application

This module contains the main FastAPI application for the SQL Agent API,
including middleware, CORS configuration, and health checks.
"""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import structlog

from sql_agent.core.config import settings
from sql_agent.core.database import db_manager
from sql_agent.agents.orchestrator import AgentOrchestrator
from sql_agent.mcp.server import mcp_server as mcp_fastapi_app
from .routes import query, sql, analysis, viz, schema

# Configure structured logging
logger = structlog.get_logger(__name__)

# Global instances
database_manager = None
orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global database_manager, orchestrator
    
    # Startup
    logger.info("Starting SQL Agent API")
    
    try:
        # Initialize database manager
        database_manager = db_manager
        await database_manager.initialize()  # <-- Ensure async engine is initialized
        
        # Initialize agent orchestrator
        try:
            orchestrator = AgentOrchestrator()
            await orchestrator.initialize()
            logger.info("Agent orchestrator initialized")
        except Exception as e:
            logger.warning(f"Agent orchestrator initialization failed: {e}")
            orchestrator = None
        
        logger.info("SQL Agent API startup complete")
        yield
        
    except Exception as e:
        logger.error("Failed to start SQL Agent API", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down SQL Agent API")
        
        # MCP server's lifespan context handles db_manager close
        
        logger.info("SQL Agent API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="SQL Agent API",
    description="AI-powered SQL Agent with natural language to SQL conversion, analysis, and visualization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Mount the FastMCP application
app.mount("/mcp", mcp_fastapi_app)

# Temporarily disable CORS and TrustedHost middleware for debugging
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.trustedhost import TrustedHostMiddleware

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # TODO: Configure from settings
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=["*"]  # TODO: Configure from settings
# )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID for tracing."""
    request_id = request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000)}")
    request.state.request_id = request_id
    
    # Add request ID to logger context
    logger = structlog.get_logger(__name__).bind(request_id=request_id)
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured logging."""
    logger.error(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "status_code": exc.status_code,
                "detail": exc.detail,
                "request_id": getattr(request.state, "request_id", "unknown")
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with structured logging."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
        method=request.method,
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_error",
                "detail": "An internal server error occurred",
                "request_id": getattr(request.state, "request_id", "unknown")
            }
        }
    )


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "services": {}
    }
    
    # Check database connection
    try:
        if database_manager:
            await database_manager.test_connection()
            health_status["services"]["database"] = "healthy"
        else:
            health_status["services"]["database"] = "not_initialized"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check MCP server
    try:
        # FastMCP doesn't have a direct health check method like the old MCPServer
        # We can check if it's initialized by checking if mcp_fastapi_app is not None
        if mcp_fastapi_app:
            health_status["services"]["mcp_server"] = "initialized"
        else:
            health_status["services"]["mcp_server"] = "not_initialized"
    except Exception as e:
        health_status["services"]["mcp_server"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check orchestrator
    try:
        if orchestrator:
            health_status["services"]["orchestrator"] = "initialized"
        else:
            health_status["services"]["orchestrator"] = "not_initialized"
    except Exception as e:
        health_status["services"]["orchestrator"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "SQL Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Import and include routers
app.include_router(query.router, prefix="/api/v1", tags=["query"])
app.include_router(sql.router, prefix="/api/v1/sql", tags=["sql"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(viz.router, prefix="/api/v1/visualization", tags=["visualization"])
app.include_router(schema.router, prefix="/api/v1/schema", tags=["schema"])