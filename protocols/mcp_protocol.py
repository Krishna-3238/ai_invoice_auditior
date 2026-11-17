"""
Model Context Protocol (MCP) - Tool Registry & Discovery
=========================================================
Implements MCP for dynamic tool discovery, access, and invocation.

Instead of hard-coding tool access within agents, MCP enables agents to
dynamically discover and invoke the tools they need through standardized
interfaces.

Key Components:
1. MCPTool - Base class for MCP-compatible tools
2. MCPToolRegistry - Central tool registry for discovery
3. MCPToolServer - Serves tools to agents via standardized interface
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("MCPProtocol")




class ToolCategory(str, Enum):
    """Categories of MCP tools."""

    MONITORING = "monitoring"
    EXTRACTION = "extraction"
    TRANSLATION = "translation"
    VALIDATION = "validation"
    REPORTING = "reporting"
    RAG = "rag"
    GUARDRAILS = "guardrails"


@dataclass
class MCPToolDescriptor:
    """
    Descriptor for an MCP-compatible tool.

    Each tool publishes its descriptor for agents to discover
    its capabilities, input/output schemas, and invocation method.
    """

    tool_id: str
    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    required_permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize descriptor."""
        d = asdict(self)
        d["category"] = self.category.value
        return d


@dataclass
class MCPToolInvocation:
    """Record of a tool invocation."""

    invocation_id: str
    tool_id: str
    agent_id: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    duration_ms: Optional[float] = None




class MCPToolRegistry:
    """
    Central registry for MCP-compatible tools.

    Agents can discover tools by name, category, or capability,
    and invoke them through a standardized interface.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}                                
            cls._instance._handlers = {}                       
            cls._instance._invocation_log = []                           
        return cls._instance

    def register_tool(
        self,
        descriptor: MCPToolDescriptor,
        handler: Callable,
    ):
        """
        Register a tool with its descriptor and handler function.

        Args:
            descriptor: Tool descriptor with metadata
            handler: Callable that implements the tool logic
        """
        self._tools[descriptor.tool_id] = descriptor
        self._handlers[descriptor.tool_id] = handler
        logger.info(
            f"MCP: Registered tool '{descriptor.name}' "
            f"({descriptor.category.value}) as '{descriptor.tool_id}'"
        )

    def deregister_tool(self, tool_id: str):
        """Remove a tool from the registry."""
        self._tools.pop(tool_id, None)
        self._handlers.pop(tool_id, None)

    def discover_tool(self, tool_id: str) -> Optional[MCPToolDescriptor]:
        """Discover a tool by ID."""
        return self._tools.get(tool_id)

    def discover_by_category(self, category: ToolCategory) -> List[MCPToolDescriptor]:
        """Discover tools by category."""
        return [desc for desc in self._tools.values() if desc.category == category]

    def discover_by_name(self, name: str) -> Optional[MCPToolDescriptor]:
        """Discover a tool by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for desc in self._tools.values():
            if name_lower in desc.name.lower():
                return desc
        return None

    def list_tools(self) -> List[MCPToolDescriptor]:
        """List all registered tools."""
        return list(self._tools.values())

    def invoke_tool(
        self,
        tool_id: str,
        agent_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Invoke a tool through the MCP interface.

        Args:
            tool_id: ID of the tool to invoke
            agent_id: ID of the calling agent
            input_data: Input data for the tool

        Returns:
            Tool output
        """
        descriptor = self._tools.get(tool_id)
        if not descriptor:
            raise ValueError(f"MCP: Tool '{tool_id}' not found in registry")

        handler = self._handlers.get(tool_id)
        if not handler:
            raise ValueError(f"MCP: No handler for tool '{tool_id}'")

        invocation = MCPToolInvocation(
            invocation_id=f"inv_{uuid.uuid4().hex[:12]}",
            tool_id=tool_id,
            agent_id=agent_id,
            input_data=input_data,
        )

        logger.info(
            f"MCP: Agent '{agent_id}' invoking tool '{descriptor.name}' "
            f"[{invocation.invocation_id}]"
        )

        start_time = datetime.now()

        try:
            result = handler(input_data)

            invocation.output_data = result
            invocation.success = True
            invocation.completed_at = datetime.now().isoformat()
            invocation.duration_ms = (
                datetime.now() - start_time
            ).total_seconds() * 1000

            logger.info(
                f"MCP: Tool '{descriptor.name}' completed in "
                f"{invocation.duration_ms:.1f}ms [{invocation.invocation_id}]"
            )

            self._invocation_log.append(invocation)
            return result

        except Exception as e:
            invocation.error = str(e)
            invocation.completed_at = datetime.now().isoformat()
            invocation.duration_ms = (
                datetime.now() - start_time
            ).total_seconds() * 1000

            logger.error(
                f"MCP: Tool '{descriptor.name}' failed: {e} "
                f"[{invocation.invocation_id}]"
            )

            self._invocation_log.append(invocation)
            raise

    def get_invocation_log(
        self, tool_id: Optional[str] = None, limit: int = 100
    ) -> List[MCPToolInvocation]:
        """Get invocation history, optionally filtered by tool."""
        log = self._invocation_log
        if tool_id:
            log = [inv for inv in log if inv.tool_id == tool_id]
        return log[-limit:]

    def reset(self):
        """Reset the registry (for testing)."""
        self._tools.clear()
        self._handlers.clear()
        self._invocation_log.clear()




def mcp_tool(
    name: str,
    description: str,
    category: ToolCategory,
    tool_id: Optional[str] = None,
    input_schema: Optional[Dict[str, Any]] = None,
    output_schema: Optional[Dict[str, Any]] = None,
):
    """
    Decorator to register a function as an MCP tool.

    Usage:
        @mcp_tool(
            name="Invoice-Watcher Tool",
            description="Monitors for new invoices",
            category=ToolCategory.MONITORING,
        )
        def watch_invoices(input_data: dict) -> dict:
            ...
    """

    def decorator(func: Callable):
        descriptor = MCPToolDescriptor(
            tool_id=tool_id or f"mcp_{func.__name__}",
            name=name,
            description=description,
            category=category,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
        )

        registry = get_tool_registry()
        registry.register_tool(descriptor, func)

        func._mcp_descriptor = descriptor
        return func

    return decorator



_tool_registry = MCPToolRegistry()


def get_tool_registry() -> MCPToolRegistry:
    """Get the global MCP tool registry."""
    return _tool_registry
