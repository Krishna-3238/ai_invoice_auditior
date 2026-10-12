CREATE SCHEMA IF NOT EXISTS erp;
CREATE TABLE IF NOT EXISTS erp.vendors(vendor_id text PRIMARY KEY,vendor_name text NOT NULL,country text,currency char(3));
CREATE TABLE IF NOT EXISTS erp.sku_master(item_code text PRIMARY KEY,category text,uom text,gst_rate numeric);
CREATE TABLE IF NOT EXISTS erp.purchase_orders(po_number text PRIMARY KEY,vendor_id text NOT NULL REFERENCES erp.vendors(vendor_id),currency char(3));
CREATE TABLE IF NOT EXISTS erp.po_line_items(id bigserial PRIMARY KEY,po_number text NOT NULL REFERENCES erp.purchase_orders(po_number) ON DELETE CASCADE,item_code text NOT NULL,description text,quantity numeric,unit_price numeric,currency char(3));
