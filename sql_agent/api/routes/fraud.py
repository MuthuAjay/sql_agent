"""
Fraud Detection Routes

This module contains endpoints for fraud detection and analysis.
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
import structlog

from sql_agent.api.dependencies import FraudDetectorsDep, FraudReportGeneratorDep, DatabaseManagerDep
from sql_agent.fraud.models import FraudDetectionRequest, FraudDetectionResponse, FraudAnalysisReport
from sql_agent.core.config import settings

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_table_for_fraud(
    table_name: str = Query(..., description="Name of the table to analyze"),
    analysis_mode: Optional[str] = Query(None, description="Analysis mode: quick, standard, or deep"),
    req: Request = None,
    fraud_detectors: Dict[str, Any] = Depends(FraudDetectorsDep),
    fraud_report_generator = Depends(FraudReportGeneratorDep),
    database_manager = Depends(DatabaseManagerDep)
) -> Dict[str, Any]:
    """
    Analyze a single table for fraud patterns and vulnerabilities.

    This endpoint runs comprehensive fraud detection analysis on the specified table.
    """
    start_time = time.time()
    request_id = getattr(req.state, "request_id", "unknown") if req else "unknown"

    logger.info(
        "Starting fraud analysis",
        request_id=request_id,
        table_name=table_name,
        analysis_mode=analysis_mode or settings.fraud_analysis_mode
    )

    try:
        # Verify table exists
        schema_info = await database_manager.get_schema_info()
        tables = [t["name"] for t in schema_info.get("tables", [])]

        if table_name not in tables:
            raise HTTPException(
                status_code=404,
                detail=f"Table '{table_name}' not found in database"
            )

        # Run all fraud detectors
        detection_results = {}

        # Transaction fraud detection
        try:
            transaction_result = await fraud_detectors['transaction'].detect(
                table_name=table_name,
                database_manager=database_manager
            )
            detection_results['transaction'] = transaction_result
        except Exception as e:
            logger.warning(f"Transaction detection failed: {e}")
            detection_results['transaction'] = None

        # Schema vulnerability detection
        try:
            schema_result = await fraud_detectors['schema'].detect(
                table_name=table_name,
                database_manager=database_manager
            )
            detection_results['schema'] = schema_result
        except Exception as e:
            logger.warning(f"Schema detection failed: {e}")
            detection_results['schema'] = None

        # Temporal anomaly detection
        try:
            temporal_result = await fraud_detectors['temporal'].detect(
                table_name=table_name,
                database_manager=database_manager
            )
            detection_results['temporal'] = temporal_result
        except Exception as e:
            logger.warning(f"Temporal detection failed: {e}")
            detection_results['temporal'] = None

        # Statistical anomaly detection
        try:
            statistical_result = await fraud_detectors['statistical'].detect(
                table_name=table_name,
                database_manager=database_manager
            )
            detection_results['statistical'] = statistical_result
        except Exception as e:
            logger.warning(f"Statistical detection failed: {e}")
            detection_results['statistical'] = None

        # Relationship integrity detection
        try:
            relationship_result = await fraud_detectors['relationship'].detect(
                table_name=table_name,
                database_manager=database_manager
            )
            detection_results['relationship'] = relationship_result
        except Exception as e:
            logger.warning(f"Relationship detection failed: {e}")
            detection_results['relationship'] = None

        processing_time = time.time() - start_time

        # Generate summary
        total_scenarios = sum(
            len(result.get('scenarios', [])) if result else 0
            for result in detection_results.values()
        )

        highest_risk = "low"
        for result in detection_results.values():
            if result and result.get('scenarios'):
                for scenario in result['scenarios']:
                    risk = scenario.get('risk_level', 'low')
                    if risk == "critical":
                        highest_risk = "critical"
                        break
                    elif risk == "high" and highest_risk != "critical":
                        highest_risk = "high"
                    elif risk == "medium" and highest_risk not in ["critical", "high"]:
                        highest_risk = "medium"

        logger.info(
            "Fraud analysis completed",
            request_id=request_id,
            table_name=table_name,
            total_scenarios=total_scenarios,
            highest_risk=highest_risk,
            processing_time=processing_time
        )

        return {
            "table_name": table_name,
            "analysis_mode": analysis_mode or settings.fraud_analysis_mode,
            "detection_results": detection_results,
            "summary": {
                "total_scenarios_detected": total_scenarios,
                "highest_risk_level": highest_risk,
                "detectors_run": len([r for r in detection_results.values() if r is not None]),
                "detectors_failed": len([r for r in detection_results.values() if r is None])
            },
            "processing_time": processing_time,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Fraud analysis failed",
            request_id=request_id,
            table_name=table_name,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Fraud analysis failed: {str(e)}"
        )


@router.post("/analyze-multi")
async def analyze_multiple_tables(
    table_names: List[str] = Query(..., description="Names of tables to analyze"),
    analysis_mode: Optional[str] = Query(None, description="Analysis mode"),
    req: Request = None,
    fraud_detectors: Dict[str, Any] = Depends(FraudDetectorsDep),
    database_manager = Depends(DatabaseManagerDep)
) -> Dict[str, Any]:
    """
    Analyze multiple tables for fraud patterns.

    This endpoint runs fraud detection on multiple tables in parallel.
    """
    start_time = time.time()
    request_id = getattr(req.state, "request_id", "unknown") if req else "unknown"

    logger.info(
        "Starting multi-table fraud analysis",
        request_id=request_id,
        table_count=len(table_names)
    )

    try:
        results = {}

        for table_name in table_names:
            try:
                # Simplified analysis for multiple tables
                transaction_result = await fraud_detectors['transaction'].detect(
                    table_name=table_name,
                    database_manager=database_manager
                )
                results[table_name] = {
                    "status": "success",
                    "scenarios_detected": len(transaction_result.get('scenarios', [])) if transaction_result else 0,
                    "result": transaction_result
                }
            except Exception as e:
                logger.warning(f"Analysis failed for {table_name}: {e}")
                results[table_name] = {
                    "status": "failed",
                    "error": str(e)
                }

        processing_time = time.time() - start_time

        return {
            "tables_analyzed": len(table_names),
            "results": results,
            "processing_time": processing_time,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id
        }

    except Exception as e:
        logger.error(
            "Multi-table fraud analysis failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Multi-table analysis failed: {str(e)}"
        )


@router.get("/scenarios")
async def get_fraud_scenarios() -> Dict[str, Any]:
    """
    Get available fraud scenario templates.

    This endpoint returns information about fraud patterns that can be detected.
    """
    try:
        from sql_agent.fraud.patterns import FraudPatternLibrary

        pattern_library = FraudPatternLibrary()
        scenarios = pattern_library.get_all_patterns()

        return {
            "scenarios": scenarios,
            "total_scenarios": len(scenarios),
            "categories": list(set(s.get('category', 'unknown') for s in scenarios))
        }

    except Exception as e:
        logger.error("Failed to get fraud scenarios", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get fraud scenarios: {str(e)}"
        )


@router.post("/report/generate")
async def generate_fraud_report(
    table_name: str = Query(..., description="Table name"),
    format: str = Query("html", description="Report format: html, json, or text"),
    fraud_detectors: Dict[str, Any] = Depends(FraudDetectorsDep),
    fraud_report_generator = Depends(FraudReportGeneratorDep),
    database_manager = Depends(DatabaseManagerDep)
) -> Any:
    """
    Generate a comprehensive fraud detection report.

    This endpoint generates a detailed report of fraud analysis results.
    """
    try:
        # Run fraud detection
        detection_results = {}

        for detector_name, detector in fraud_detectors.items():
            try:
                result = await detector.detect(
                    table_name=table_name,
                    database_manager=database_manager
                )
                detection_results[detector_name] = result
            except Exception as e:
                logger.warning(f"Detector {detector_name} failed: {e}")

        # Generate report
        if format == "html":
            report = fraud_report_generator.generate_html_report(detection_results, table_name)
            return HTMLResponse(content=report)
        elif format == "json":
            report = fraud_report_generator.generate_json_report(detection_results, table_name)
            return JSONResponse(content=report)
        elif format == "text":
            report = fraud_report_generator.generate_text_report(detection_results, table_name)
            return {"report": report, "format": "text"}
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {format}. Use 'html', 'json', or 'text'"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate fraud report", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate fraud report: {str(e)}"
        )


@router.get("/health")
async def fraud_service_health(
    fraud_detectors: Dict[str, Any] = Depends(FraudDetectorsDep)
) -> Dict[str, Any]:
    """
    Health check for fraud detection service.

    Returns the status of all fraud detectors.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "detectors": {
            name: "available" for name in fraud_detectors.keys()
        },
        "config": {
            "enabled": settings.enable_fraud_detection,
            "analysis_mode": settings.fraud_analysis_mode,
            "timeout": settings.fraud_detection_timeout,
            "confidence_threshold": settings.fraud_confidence_threshold,
            "risk_threshold": settings.fraud_risk_level_threshold
        }
    }
