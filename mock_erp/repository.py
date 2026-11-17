"""PostgreSQL repository for ERP master data used by invoice validation."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import psycopg
from psycopg.types.json import Jsonb
from integrations.storage import RedisCache


class ERPRepository:
    """Stores ERP vendors, purchase orders, and SKUs in PostgreSQL."""

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.getenv("DATABASE_URL")
        if not self.dsn:
            raise RuntimeError("DATABASE_URL must be set to connect to the ERP database")
        self.cache = RedisCache(ttl_seconds=int(os.getenv("ERP_CACHE_TTL_SECONDS", "3600")))

    def _lookup(self, kind: str, key: str, query: str):
        cache_key = f"invoice-auditor:erp:{kind}:{key}"
        if self.cache.enabled:
            try:
                cached = self.cache.get_json(cache_key)
                if cached is not None:
                    return cached
            except Exception:
                pass  # Cache failures must not block authoritative database lookups.
        with psycopg.connect(self.dsn) as conn:
            row = conn.execute(query, (key,)).fetchone()
        value = row[0] if row else None
        if value is not None and self.cache.enabled:
            try:
                self.cache.set_json(cache_key, value)
            except Exception:
                pass
        return value

    def seed_from_json(self, data_dir: Path) -> Dict[str, int]:
        """Idempotently upsert the supplied mock ERP master data into PostgreSQL."""
        with open(data_dir / "vendors.json", encoding="utf-8") as handle:
            vendors = json.load(handle)
        with open(data_dir / "PO Records.json", encoding="utf-8") as handle:
            purchase_orders = json.load(handle)
        with open(data_dir / "sku_master.json", encoding="utf-8") as handle:
            skus = json.load(handle)

        with psycopg.connect(self.dsn) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS erp_vendors (
                vendor_id TEXT PRIMARY KEY, vendor_name TEXT NOT NULL,
                country TEXT, currency TEXT, record JSONB NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS erp_purchase_orders (
                po_number TEXT PRIMARY KEY,
                vendor_id TEXT NOT NULL REFERENCES erp_vendors(vendor_id),
                line_items JSONB NOT NULL, record JSONB NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS erp_skus (
                item_code TEXT PRIMARY KEY, category TEXT, uom TEXT,
                gst_rate NUMERIC, record JSONB NOT NULL)""")

            for vendor in vendors:
                conn.execute(
                    """INSERT INTO erp_vendors
                       (vendor_id, vendor_name, country, currency, record)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (vendor_id) DO UPDATE SET
                       vendor_name=EXCLUDED.vendor_name, country=EXCLUDED.country,
                       currency=EXCLUDED.currency, record=EXCLUDED.record""",
                    (vendor["vendor_id"], vendor["vendor_name"], vendor.get("country"),
                     vendor.get("currency"), Jsonb(vendor)),
                )

            for po in purchase_orders:
                conn.execute(
                    """INSERT INTO erp_purchase_orders
                       (po_number, vendor_id, line_items, record)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (po_number) DO UPDATE SET
                       vendor_id=EXCLUDED.vendor_id, line_items=EXCLUDED.line_items,
                       record=EXCLUDED.record""",
                    (po["po_number"], po["vendor_id"], Jsonb(po.get("line_items", [])), Jsonb(po)),
                )

            for sku in skus:
                conn.execute(
                    """INSERT INTO erp_skus (item_code, category, uom, gst_rate, record)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (item_code) DO UPDATE SET
                       category=EXCLUDED.category, uom=EXCLUDED.uom,
                       gst_rate=EXCLUDED.gst_rate, record=EXCLUDED.record""",
                    (sku["item_code"], sku.get("category"), sku.get("uom"),
                     sku.get("gst_rate"), Jsonb(sku)),
                )

        # Refresh cached master records after idempotent fixture upserts.
        for kind, records, key_name in (
            ("vendor", vendors, "vendor_id"),
            ("purchase_order", purchase_orders, "po_number"),
            ("sku", skus, "item_code"),
        ):
            for record in records:
                try:
                    self.cache.delete(f"invoice-auditor:erp:{kind}:{record[key_name]}")
                except Exception:
                    pass

        return {"vendors": len(vendors), "purchase_orders": len(purchase_orders), "skus": len(skus)}

    def counts(self) -> Dict[str, int]:
        with psycopg.connect(self.dsn) as conn:
            return {
                "vendors": conn.execute("SELECT COUNT(*) FROM erp_vendors").fetchone()[0],
                "purchase_orders": conn.execute("SELECT COUNT(*) FROM erp_purchase_orders").fetchone()[0],
                "skus": conn.execute("SELECT COUNT(*) FROM erp_skus").fetchone()[0],
            }

    def get_vendor(self, vendor_id: str) -> Optional[Dict[str, Any]]:
        return self._lookup("vendor", vendor_id,
                            "SELECT record FROM erp_vendors WHERE vendor_id=%s")

    def get_purchase_order(self, po_number: str) -> Optional[Dict[str, Any]]:
        return self._lookup("purchase_order", po_number,
                            "SELECT record FROM erp_purchase_orders WHERE po_number=%s")

    def get_sku(self, item_code: str) -> Optional[Dict[str, Any]]:
        return self._lookup("sku", item_code,
                            "SELECT record FROM erp_skus WHERE item_code=%s")
