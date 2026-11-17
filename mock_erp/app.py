
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging
from pathlib import Path

from mock_erp.repository import ERPRepository

logger = logging.getLogger("MockERP")

app = FastAPI(
    title="Mock ERP System",
    description="Mock ERP API for AI Invoice Auditor",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "ERP_mockdata"
erp_repository = ERPRepository()


def load_erp_data():
    """Seed master data into PostgreSQL; runtime lookup is database-backed."""
    counts = erp_repository.seed_from_json(DATA_DIR)
    logger.info("Seeded ERP database from mock master files: %s", counts)
    return counts


@app.on_event("startup")
async def startup_event():
    """Upsert source fixtures into PostgreSQL when the API starts."""
    load_erp_data()
    logger.info("Mock ERP System ready")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    counts = erp_repository.counts()
    return {
        "status": "healthy",
        "vendors_loaded": counts["vendors"],
        "po_records_loaded": counts["purchase_orders"],
        "sku_records_loaded": counts["skus"],
    }


@app.get("/vendor/{vendor_id}")
async def get_vendor(vendor_id: str):
    """
    Get vendor information by vendor ID

    Args:
        vendor_id: Vendor identifier (e.g., VEND-001)

    Returns:
        Vendor details including name, country, and default currency
    """
    vendor = erp_repository.get_vendor(vendor_id)

    if not vendor:
        raise HTTPException(
            status_code=404, detail=f"Vendor '{vendor_id}' not found in ERP system"
        )

    return {"success": True, "data": vendor}


@app.get("/po/{po_number}")
async def get_po(po_number: str):
    """
    Get Purchase Order details by PO number

    Args:
        po_number: Purchase Order number (e.g., PO-1001)

    Returns:
        PO details including vendor, line items, quantities, and prices
    """
    po = erp_repository.get_purchase_order(po_number)

    if not po:
        raise HTTPException(
            status_code=404,
            detail=f"Purchase Order '{po_number}' not found in ERP system",
        )

    return {"success": True, "data": po}


@app.get("/sku/{item_code}")
async def get_sku(item_code: str):
    """
    Get SKU/Item master data by item code

    Args:
        item_code: Item/SKU code (e.g., SKU-001)

    Returns:
        SKU details including category, unit of measure, and GST rate
    """
    sku = erp_repository.get_sku(item_code)

    if not sku:
        raise HTTPException(
            status_code=404, detail=f"SKU '{item_code}' not found in SKU master"
        )

    return {"success": True, "data": sku}


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Resource not found",
            "detail": str(exc.detail) if hasattr(exc, "detail") else "Not found",
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
