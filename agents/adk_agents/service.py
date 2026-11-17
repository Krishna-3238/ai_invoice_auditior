"""HTTP A2A and MCP server for one independently deployed ADK agent."""

import os
import json
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from protocols.mcp_protocol import get_tool_registry
from protocols.a2a_protocol import get_registry
from config.logging_config import setup_logging
from observability import get_observability

setup_logging()

app = FastAPI(title="Invoice Auditor Agent Service", version="1.0.0")
agent = None
TOOL_IDS = {
    "extractor": "data_harvester_tool",
    "translator": "lang_bridge_tool",
    "reporter": "insight_reporter_tool",
}


def _load_agent():
    kind = os.getenv("AGENT_TYPE", "").lower()
    if kind == "extractor":
        from agents.adk_agents.extractor_agent import ADKExtractorAgent
        return ADKExtractorAgent()
    if kind == "translator":
        from agents.adk_agents.translation_agent import ADKTranslationAgent
        return ADKTranslationAgent()
    if kind == "reporter":
        from agents.adk_agents.reporting_agent import ADKReportingAgent
        return ADKReportingAgent()
    raise RuntimeError("AGENT_TYPE must be extractor, translator, or reporter")


@app.on_event("startup")
async def startup():
    global agent
    agent = _load_agent()


@app.get("/health")
def health():
    return {"status": "healthy", "agent_id": getattr(agent, "agent_id", None)}


@app.get("/agent-card")
def agent_card():
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent is not initialized")
    card = get_registry().discover(agent.agent_id)
    return card.to_dict() if card else {"agent_id": agent.agent_id}


@app.post("/invoke")
def invoke(message: Dict[str, Any]):
    """Accept the project's A2A message envelope and return a response envelope."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent is not initialized")
    payload = message.get("payload", message)
    try:
        trace_id = payload.get("_trace_id")
        if trace_id:
            get_observability().attach_trace(str(trace_id), name="invoice_audit")
        # A2A carries the request across the network; dispatch inside the
        # service through the agent's registered MCP tool.
        tool_id = TOOL_IDS[os.getenv("AGENT_TYPE", "").lower()]
        mcp_url = os.getenv("MCP_SERVER_URL", "").rstrip("/")
        if mcp_url:
            import requests
            response = requests.post(
                mcp_url,
                json={
                    "jsonrpc": "2.0", "id": uuid.uuid4().hex,
                    "method": "tools/call",
                    "params": {"name": tool_id, "arguments": payload},
                },
                timeout=float(os.getenv("MCP_REQUEST_TIMEOUT_SECONDS", "300")),
            )
            response.raise_for_status()
            rpc = response.json()
            if "error" in rpc:
                raise RuntimeError(rpc["error"].get("message", "MCP tool call failed"))
            content = rpc.get("result", {}).get("content", [])
            result = json.loads(content[0]["text"]) if content else {}
        else:
            # Allows an individual agent service to be developed locally without
            # the compose MCP gateway; the deployed compose path uses HTTP MCP.
            result = get_tool_registry().invoke_tool(
                tool_id, message.get("source_agent", "a2a_client"), payload
            )
        return {
            "message_id": f"msg_{os.urandom(8).hex()}",
            "message_type": "response",
            "source_agent": agent.agent_id,
            "target_agent": message.get("source_agent", "unknown"),
            "correlation_id": message.get("correlation_id"),
            "payload": result,
        }
    except Exception as exc:
        return {
            "message_id": f"msg_{os.urandom(8).hex()}",
            "message_type": "error",
            "source_agent": agent.agent_id,
            "target_agent": message.get("source_agent", "unknown"),
            "correlation_id": message.get("correlation_id"),
            "payload": {},
            "error": str(exc),
        }


@app.get("/mcp/tools")
def list_tools():
    """Expose tools registered by this agent to MCP-compatible clients."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent is not initialized")
    return {"tools": [tool.to_dict() for tool in get_tool_registry().list_tools()]}


@app.post("/mcp/invoke")
def invoke_mcp_tool(request: Dict[str, Any]):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent is not initialized")
    try:
        trace_id = request.get("input_data", {}).get("_trace_id")
        if trace_id:
            get_observability().attach_trace(str(trace_id), name="invoice_audit")
        result = get_tool_registry().invoke_tool(
            request["tool_id"], request.get("agent_id", "mcp_client"),
            request.get("input_data", {}),
        )
        return {"success": True, "result": result}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
