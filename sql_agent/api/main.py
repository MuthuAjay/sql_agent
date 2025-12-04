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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import structlog

from sql_agent.core.config import settings
from sql_agent.core.database import db_manager
from sql_agent.agents.orchestrator import AgentOrchestrator
from sql_agent.mcp.server import mcp_server as mcp_fastapi_app
from sql_agent.api.dependencies import set_global_instances, set_fraud_instances
from sql_agent.api.models import HealthCheckResponse, ErrorResponse
from .routes import query, sql, analysis, viz, schema

# Import fraud detection components
try:
    from sql_agent.fraud.detectors.transaction import TransactionFraudDetector
    from sql_agent.fraud.detectors.schema import SchemaVulnerabilityDetector
    from sql_agent.fraud.detectors.temporal import TemporalAnomalyDetector
    from sql_agent.fraud.detectors.statistical import StatisticalAnomalyDetector
    from sql_agent.fraud.detectors.relationship import RelationshipIntegrityDetector
    from sql_agent.fraud.reporting import FraudReportGenerator
    FRAUD_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Fraud detection modules not available: {e}")
    FRAUD_AVAILABLE = False

# Configure structured logging
logger = structlog.get_logger(__name__)

# Global instances
database_manager = None
orchestrator = None
fraud_detectors = None
fraud_report_generator = None

"""
RAG Initialization Fix

Add this to your main.py startup to properly initialize RAG components.
"""

# Add this to your sql_agent/api/main.py in the lifespan function:

async def initialize_rag_components():
    """Initialize RAG components with proper error handling."""
    logger.info("Initializing RAG components")

    try:
        # Initialize context manager
        from sql_agent.rag.context import context_manager
        await context_manager.initialize()
        logger.info("RAG context manager initialized successfully")

        return True

    except Exception as e:
        logger.warning(f"RAG initialization failed (will use fallback): {e}")
        # Don't fail startup - RAG is optional
        return False

async def initialize_fraud_detectors():
    """Initialize fraud detection components."""
    logger.info("Initializing fraud detection components")

    try:
        if not FRAUD_AVAILABLE:
            logger.info("Fraud detection modules not available")
            return None, None

        if not settings.enable_fraud_detection:
            logger.info("Fraud detection disabled in configuration")
            return None, None

        # Get LLM provider from orchestrator
        llm_provider = orchestrator.llm_provider if orchestrator else None

        # Initialize all fraud detectors
        detectors = {
            'transaction': TransactionFraudDetector(llm_provider=llm_provider),
            'schema': SchemaVulnerabilityDetector(llm_provider=llm_provider),
            'temporal': TemporalAnomalyDetector(llm_provider=llm_provider),
            'statistical': StatisticalAnomalyDetector(llm_provider=llm_provider),
            'relationship': RelationshipIntegrityDetector(llm_provider=llm_provider)
        }

        # Initialize fraud report generator
        report_gen = FraudReportGenerator()

        logger.info("Fraud detectors initialized successfully")
        return detectors, report_gen

    except Exception as e:
        logger.warning(f"Fraud detection initialization failed (will use fallback): {e}")
        return None, None

# Modify your lifespan function in main.py:

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global database_manager, orchestrator, fraud_detectors, fraud_report_generator

    # Startup
    logger.info("Starting SQL Agent API")

    try:
        # Initialize database manager
        database_manager = db_manager
        await database_manager.initialize()
        logger.info("Database manager initialized")

        # Initialize agent orchestrator
        try:
            orchestrator = AgentOrchestrator()
            await orchestrator.initialize()
            logger.info("Agent orchestrator initialized")
        except Exception as e:
            logger.warning(f"Agent orchestrator initialization failed: {e}")
            orchestrator = None

        # ADDED: Initialize RAG components
        rag_initialized = await initialize_rag_components()
        if rag_initialized:
            logger.info("RAG components initialized successfully")
        else:
            logger.info("RAG components unavailable - using fallback mode")

        # ADDED: Initialize fraud detection components
        fraud_detectors, fraud_report_generator = await initialize_fraud_detectors()
        if fraud_detectors:
            set_fraud_instances(fraud_detectors, fraud_report_generator)
            logger.info("Fraud detection components initialized successfully")
        else:
            logger.info("Fraud detection components unavailable")

        # Set global instances for dependency injection
        set_global_instances(database_manager, orchestrator)
        logger.info("Dependencies configured")

        logger.info("SQL Agent API startup complete")
        yield

    except Exception as e:
        logger.error("Failed to start SQL Agent API", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down SQL Agent API")

        if orchestrator:
            try:
                await orchestrator.cleanup()
                logger.info("Agent orchestrator cleaned up")
            except Exception as e:
                logger.warning(f"Error during orchestrator cleanup: {e}")

        # ADDED: Cleanup RAG components
        try:
            from sql_agent.rag.context import context_manager
            # Context manager cleanup (if it has a cleanup method)
            logger.info("RAG components cleaned up")
        except Exception as e:
            logger.warning(f"Error during RAG cleanup: {e}")

        if database_manager:
            try:
                await database_manager.close()
                logger.info("Database manager closed")
            except Exception as e:
                logger.warning(f"Error during database cleanup: {e}")

        logger.info("SQL Agent API shutdown complete")

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Application lifespan manager for startup and shutdown."""
#     global database_manager, orchestrator

#     # Startup
#     logger.info("Starting SQL Agent API")

#     try:
#         # Initialize database manager
#         database_manager = db_manager
#         await database_manager.initialize()
#         logger.info("Database manager initialized")

#         # Initialize agent orchestrator
#         try:
#             orchestrator = AgentOrchestrator()
#             await orchestrator.initialize()
#             logger.info("Agent orchestrator initialized")
#         except Exception as e:
#             logger.warning(f"Agent orchestrator initialization failed: {e}")
#             orchestrator = None

#         # Set global instances for dependency injection
#         set_global_instances(database_manager, orchestrator)
#         logger.info("Dependencies configured")

#         logger.info("SQL Agent API startup complete")
#         yield

#     except Exception as e:
#         logger.error("Failed to start SQL Agent API", error=str(e))
#         raise
#     finally:
#         # Shutdown
#         logger.info("Shutting down SQL Agent API")

#         if orchestrator:
#             try:
#                 await orchestrator.cleanup()
#                 logger.info("Agent orchestrator cleaned up")
#             except Exception as e:
#                 logger.warning(f"Error during orchestrator cleanup: {e}")

#         if database_manager:
#             try:
#                 await database_manager.close()
#                 logger.info("Database manager closed")
#             except Exception as e:
#                 logger.warning(f"Error during database cleanup: {e}")

#         logger.info("SQL Agent API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="SQL Agent API",
    description="AI-powered SQL Agent with natural language to SQL conversion, analysis, and visualization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)

# Mount the FastMCP application
app.mount("/mcp", mcp_fastapi_app)

# CORS Configuration
allowed_origins = [
    "http://localhost:3000",  # React dev server
    "http://localhost:5173",  # Vite dev server
    "http://localhost:8080",  # Alternative dev server
]

if settings.ENVIRONMENT == "development":
    allowed_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
)

# Trusted Host Configuration
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS or ["*"]
    )


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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests."""
    start_time = time.time()

    logger.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        query_params=dict(request.query_params),
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        request_id=getattr(request.state, "request_id", "unknown"),
    )

    response = await call_next(request)

    duration = time.time() - start_time

    logger.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=duration,
        request_id=getattr(request.state, "request_id", "unknown"),
    )

    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured logging."""
    logger.error(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method,
        request_id=getattr(request.state, "request_id", "unknown"),
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "status_code": exc.status_code,
                "detail": exc.detail,
                "request_id": getattr(request.state, "request_id", "unknown"),
                "timestamp": time.time(),
            }
        },
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
        request_id=getattr(request.state, "request_id", "unknown"),
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_error",
                "detail": "An internal server error occurred",
                "request_id": getattr(request.state, "request_id", "unknown"),
                "timestamp": time.time(),
            }
        },
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "services": {},
    }

    # Check database connection
    try:
        if database_manager:
            await database_manager.test_connection()
            health_status["services"]["database"] = "healthy"
        else:
            health_status["services"]["database"] = "not_initialized"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    # Check MCP server
    try:
        if mcp_fastapi_app:
            health_status["services"]["mcp_server"] = "initialized"
        else:
            health_status["services"]["mcp_server"] = "not_initialized"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["mcp_server"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    # Check orchestrator
    try:
        if orchestrator:
            # Optionally test orchestrator health
            health_status["services"]["orchestrator"] = "initialized"
        else:
            health_status["services"]["orchestrator"] = "not_initialized"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["orchestrator"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    # Check fraud detectors
    try:
        if fraud_detectors:
            health_status["services"]["fraud_detection"] = "initialized"
        else:
            health_status["services"]["fraud_detection"] = "not_initialized"
    except Exception as e:
        health_status["services"]["fraud_detection"] = f"unhealthy: {str(e)}"

    return HealthCheckResponse(**health_status)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "SQL Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "running",
    }


@app.get("/api/v1/info")
async def api_info() -> Dict[str, Any]:
    """API information endpoint."""
    return {
        "name": "SQL Agent API",
        "version": "1.0.0",
        "description": "AI-powered SQL Agent with natural language to SQL conversion",
        "features": [
            "Natural language to SQL conversion",
            "SQL query execution and validation",
            "Data analysis and profiling",
            "Visualization suggestions",
            "Schema exploration",
            "Query optimization",
        ],
        "endpoints": {
            "query": "/api/v1/query",
            "sql": "/api/v1/sql",
            "analysis": "/api/v1/analysis",
            "visualization": "/api/v1/visualization",
            "schema": "/api/v1/schema",
        },
    }


# Include routers with proper prefixes and tags
app.include_router(
    query.router,
    prefix="/api/v1/query",
    tags=["Query Processing"],
    responses={404: {"description": "Query not found"}},
)

app.include_router(
    sql.router,
    prefix="/api/v1/sql",
    tags=["SQL Execution"],
    responses={400: {"description": "Invalid SQL"}},
)

app.include_router(
    analysis.router,
    prefix="/api/v1/analysis",
    tags=["Data Analysis"],
    responses={400: {"description": "Analysis failed"}},
)

app.include_router(
    viz.router,
    prefix="/api/v1/visualization",
    tags=["Data Visualization"],
    responses={400: {"description": "Visualization failed"}},
)

app.include_router(
    schema.router,
    prefix="/api/v1/schema",
    tags=["Schema Management"],
    responses={404: {"description": "Schema not found"}},
)

# Import and include fraud router if available
if FRAUD_AVAILABLE:
    try:
        from .routes import fraud
        app.include_router(
            fraud.router,
            prefix="/api/v1/fraud",
            tags=["Fraud Detection"],
            responses={400: {"description": "Fraud analysis failed"}},
        )
        logger.info("Fraud detection routes registered")
    except ImportError as e:
        logger.warning(f"Fraud routes not available: {e}")


# Additional utility endpoints
@app.get("/api/v1/status")
async def get_status():
    """Get detailed API status."""
    return {
        "api_status": "running",
        "timestamp": time.time(),
        "uptime": time.time() - start_time if "start_time" in globals() else None,
        "database_connected": database_manager is not None,
        "orchestrator_available": orchestrator is not None,
        "mcp_server_mounted": mcp_fastapi_app is not None,
    }


@app.post("/api/v1/ping")
async def ping():
    """Simple ping endpoint for connectivity testing."""
    return {
        "message": "pong",
        "timestamp": time.time(),
        "request_id": f"ping_{int(time.time() * 1000)}",
    }


# Error handling for specific scenarios
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions."""
    logger.warning(
        "Value error", error=str(exc), path=request.url.path, method=request.method
    )

    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "type": "validation_error",
                "detail": str(exc),
                "request_id": getattr(request.state, "request_id", "unknown"),
            }
        },
    )


@app.exception_handler(TimeoutError)
async def timeout_error_handler(request: Request, exc: TimeoutError):
    """Handle timeout exceptions."""
    logger.error(
        "Request timeout", error=str(exc), path=request.url.path, method=request.method
    )

    return JSONResponse(
        status_code=408,
        content={
            "error": {
                "type": "timeout_error",
                "detail": "Request timed out",
                "request_id": getattr(request.state, "request_id", "unknown"),
            }
        },
    )


# Set startup time
start_time = time.time()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "sql_agent.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.ENVIRONMENT == "development" else False,
        log_level="info",
        access_log=True,
    )
