"""ERP service. ERP tables are owned here and accessed only through HTTP."""
from __future__ import annotations

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
import psycopg
from psycopg.rows import dict_row

from backned.config.settings import DATABASE_URL

app = FastAPI(title="Mock ERP Service")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ERP_mockdata"


def _connect():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


@app.on_event("startup")
def initialize_erp():
    # Idempotent fixture import into the ERP-owned schema.
    with _connect() as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS erp")
        conn.execute("CREATE TABLE IF NOT EXISTS erp.vendors(vendor_id text PRIMARY KEY,vendor_name text,country text,currency char(3))")
        conn.execute("CREATE TABLE IF NOT EXISTS erp.sku_master(item_code text PRIMARY KEY,category text,uom text,gst_rate numeric)")
        conn.execute("CREATE TABLE IF NOT EXISTS erp.purchase_orders(po_number text PRIMARY KEY,vendor_id text REFERENCES erp.vendors(vendor_id),currency char(3))")
        conn.execute("CREATE TABLE IF NOT EXISTS erp.po_line_items(id bigserial PRIMARY KEY,po_number text REFERENCES erp.purchase_orders(po_number) ON DELETE CASCADE,item_code text,description text,quantity numeric,unit_price numeric,currency char(3))")
        for v in _read("vendors.json"):
            conn.execute("INSERT INTO erp.vendors VALUES (%s,%s,%s,%s) ON CONFLICT(vendor_id) DO UPDATE SET vendor_name=excluded.vendor_name,country=excluded.country,currency=excluded.currency", (v["vendor_id"],v["vendor_name"],v.get("country"),v.get("currency")))
        for sku in _read("sku_master.json"):
            conn.execute("INSERT INTO erp.sku_master VALUES (%s,%s,%s,%s) ON CONFLICT(item_code) DO UPDATE SET category=excluded.category,uom=excluded.uom,gst_rate=excluded.gst_rate", (sku["item_code"],sku.get("category"),sku.get("uom"),sku.get("gst_rate")))
        for po in _read("PO Records.json"):
            currency = (po.get("line_items") or [{}])[0].get("currency")
            conn.execute("INSERT INTO erp.purchase_orders VALUES (%s,%s,%s) ON CONFLICT(po_number) DO UPDATE SET vendor_id=excluded.vendor_id,currency=excluded.currency", (po["po_number"],po["vendor_id"],currency))
            conn.execute("DELETE FROM erp.po_line_items WHERE po_number=%s", (po["po_number"],))
            for li in po.get("line_items", []):
                conn.execute("INSERT INTO erp.po_line_items(po_number,item_code,description,quantity,unit_price,currency) VALUES (%s,%s,%s,%s,%s,%s)", (po["po_number"],li.get("item_code"),li.get("description"),li.get("qty"),li.get("unit_price"),li.get("currency")))


def _read(name: str):
    path = DATA / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


@app.get("/health")
def health():
    with _connect() as conn:
        conn.execute("SELECT 1 FROM erp.vendors LIMIT 1")
    return {"status": "healthy"}


@app.get("/po/{po_number}")
def get_purchase_order(po_number: str):
    result = _po(po_number)
    if result is None:
        raise HTTPException(404, "Purchase order not found")
    return result


def _po(po_number: str):
    with _connect() as conn:
        head = conn.execute("SELECT * FROM erp.purchase_orders WHERE upper(po_number)=upper(%s)", (po_number,)).fetchone()
        if not head:
            return None
        lines = conn.execute("SELECT item_code,description,quantity AS qty,unit_price,currency FROM erp.po_line_items WHERE po_number=%s ORDER BY id", (head["po_number"],)).fetchall()
        return {**head, "line_items": lines}


@app.get("/vendor/{vendor_id}")
def get_vendor(vendor_id: str):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM erp.vendors WHERE upper(vendor_id)=upper(%s)", (vendor_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Vendor not found")
    return row


@app.get("/sku/{item_code}")
def get_sku(item_code: str):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM erp.sku_master WHERE upper(item_code)=upper(%s)", (item_code,)).fetchone()
    if not row:
        raise HTTPException(404, "SKU not found")
    return row


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backned.api.erp_mock_api:app", host="0.0.0.0", port=8000)
