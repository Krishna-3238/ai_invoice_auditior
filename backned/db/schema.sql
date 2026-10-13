CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS langgraph;
CREATE TABLE IF NOT EXISTS audit.invoice_audit (
  invoice_id UUID PRIMARY KEY,
  file_path VARCHAR(512) NOT NULL,
  file_checksum VARCHAR(64) NOT NULL UNIQUE,
  extraction_method VARCHAR(32), raw_extracted_text TEXT NOT NULL DEFAULT '',
  translated_text TEXT NOT NULL DEFAULT '', source_language VARCHAR(8),
  translation_confidence NUMERIC(4,3), vendor_name VARCHAR(255), po_number VARCHAR(64),
  invoice_number VARCHAR(128), invoice_date DATE, currency CHAR(3),
  total_amount NUMERIC(14,2), validation_status VARCHAR(32),
  recommendation VARCHAR(32), extracted_raw JSONB NOT NULL DEFAULT '{}'::jsonb,
  report_json JSONB NOT NULL DEFAULT '{}'::jsonb, report_html TEXT,
  thread_id VARCHAR(128), processing_error TEXT, processed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE audit.invoice_audit ADD COLUMN IF NOT EXISTS processing_error TEXT;
CREATE TABLE IF NOT EXISTS audit.invoice_line_items (
  id BIGSERIAL PRIMARY KEY, invoice_id UUID NOT NULL REFERENCES audit.invoice_audit(invoice_id) ON DELETE CASCADE,
  line_number INTEGER NOT NULL, item_code VARCHAR(64), description TEXT,
  quantity NUMERIC(12,3), unit_price NUMERIC(14,4), line_total NUMERIC(14,2)
);
CREATE TABLE IF NOT EXISTS audit.discrepancies (
  id BIGSERIAL PRIMARY KEY, invoice_id UUID NOT NULL REFERENCES audit.invoice_audit(invoice_id) ON DELETE CASCADE,
  field_name VARCHAR(64), invoice_value TEXT, erp_value TEXT, deviation_pct NUMERIC(8,4),
  severity VARCHAR(16), detected_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS audit.human_feedback (
  id BIGSERIAL PRIMARY KEY, invoice_id UUID NOT NULL REFERENCES audit.invoice_audit(invoice_id),
  field_name VARCHAR(64) NOT NULL, original_value TEXT, corrected_value TEXT,
  corrected_by VARCHAR(128), corrected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  revalidated BOOLEAN NOT NULL DEFAULT FALSE, revalidation_result JSONB
);
CREATE INDEX IF NOT EXISTS invoice_audit_processed_at_idx ON audit.invoice_audit(processed_at DESC);
CREATE INDEX IF NOT EXISTS invoice_line_items_invoice_idx ON audit.invoice_line_items(invoice_id);
CREATE INDEX IF NOT EXISTS discrepancies_invoice_idx ON audit.discrepancies(invoice_id);
