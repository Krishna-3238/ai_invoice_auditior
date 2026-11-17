"""HTTP JSON-RPC gateway for tools hosted by the ADK agent services."""

import os
import json
from typing import Any, Dict

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from config.logging_config import setup_logging

setup_logging()

app = FastAPI(title="Invoice Auditor MCP Gateway", version="1.0.0")
AGENTS = {
    "extractor": {"url_env": "A2A_EXTRACTOR_URL", "tool_id": "data_harvester_tool"},
    "translator": {"url_env": "A2A_TRANSLATOR_URL", "tool_id": "lang_bridge_tool"},
    "reporter": {"url_env": "A2A_REPORTER_URL", "tool_id": "insight_reporter_tool"},
}


def _agent_url(kind: str) -> str:
    value = os.getenv(AGENTS[kind]["url_env"], "").rstrip("/")
    if not value:
        raise RuntimeError(f"{AGENTS[kind]['url_env']} is not configured")
    return value


def _rpc_error(request_id, code: int, message: str):
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message}}


@app.get("/health")
def health():
    return {"status": "healthy", "protocol": "MCP JSON-RPC"}


@app.post("/mcp")
def handle_mcp(rpc: Dict[str, Any]):
    request_id = rpc.get("id")
    method = rpc.get("method")
    params = rpc.get("params", {})
    try:
        if method == "initialize":
            result = {
                "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "invoice-auditor-mcp", "version": "1.0.0"},
            }
        elif method == "notifications/initialized":
            return JSONResponse(status_code=202, content={})
        elif method == "tools/list":
            tools = []
            for kind in AGENTS:
                response = requests.get(f"{_agent_url(kind)}/mcp/tools", timeout=10)
                response.raise_for_status()
                tools.extend({
                    "name": tool.get("tool_id"),
                    "description": tool.get("description", ""),
                    "inputSchema": tool.get("input_schema", {"type": "object"}),
                } for tool in response.json().get("tools", []))
            result = {"tools": tools}
        elif method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            match = next((kind for kind, config in AGENTS.items()
                          if config["tool_id"] == name), None)
            if not match:
                raise ValueError(f"Unknown MCP tool: {name}")
            response = requests.post(
                f"{_agent_url(match)}/mcp/invoke",
                json={"tool_id": name, "agent_id": "mcp_client", "input_data": arguments},
                timeout=float(os.getenv("MCP_REQUEST_TIMEOUT_SECONDS", "300")),
            )
            response.raise_for_status()
            result = {"content": [{"type": "text", "text": json.dumps(
                response.json().get("result", {}), default=str)}], "isError": False}
        else:
            return JSONResponse(_rpc_error(request_id, -32601, f"Method not found: {method}"))
        return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})
    except Exception as exc:
        return JSONResponse(_rpc_error(request_id, -32603, str(exc)), status_code=500)
