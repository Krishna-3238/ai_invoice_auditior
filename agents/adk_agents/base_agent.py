"""
Google ADK Base Agent
=====================
Base class for all Google ADK-based agents in the Invoice Auditor system.

Provides:
- Standard agent lifecycle management
- A2A protocol registration
- MCP tool registration
- RAI guardrails integration
- LangFuse observability hooks
"""

import logging
from typing import Dict, Any, Callable
from abc import ABC, abstractmethod

from protocols.a2a_protocol import (
    AgentCard,
    AgentFramework,
    AgentStatus,
    get_registry,
)
from protocols.mcp_protocol import MCPToolDescriptor, get_tool_registry
from guardrails import get_guardrails
from observability import get_observability

logger = logging.getLogger("ADKBaseAgent")


class ADKBaseAgent(ABC):
    """
    Base class for Google ADK agents.

    All ADK agents inherit from this base, which handles:
    - A2A agent card registration
    - MCP tool registration
    - Guardrails wrapping
    - Observability integration
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        capabilities: list = None,
        tools: list = None,
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
        self.tools = tools or []

        self.registry = get_registry()
        self.tool_registry = get_tool_registry()
        self.guardrails = get_guardrails()
        self.observability = get_observability()

        self._register_a2a()

        logger.info(f"ADK Agent initialized: {self.name} ({self.agent_id})")

    def _register_a2a(self):
        """Register this agent with the A2A protocol registry."""
        card = AgentCard(
            agent_id=self.agent_id,
            name=self.name,
            description=self.description,
            framework=AgentFramework.GOOGLE_ADK,
            capabilities=self.capabilities,
            tools=self.tools,
            status=AgentStatus.ACTIVE,
        )
        self.registry.register(card, handler=self._handle_a2a_request)

    def _handle_a2a_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming A2A requests.

        This wraps the agent's run() method with guardrails and observability.
        """
        trace_id = payload.get("_trace_id", "unknown")
        span_id = self.observability.start_span(
            trace_id=trace_id,
            name=f"adk_{self.agent_id}",
            input_data={"keys": list(payload.keys())},
        )

        try:
            result = self.run(payload)

            self.observability.end_span(
                span_id,
                output={
                    "keys": (
                        list(result.keys()) if isinstance(result, dict) else "non-dict"
                    )
                },
            )

            return result

        except Exception as e:
            self.observability.end_span(
                span_id,
                output={"error": str(e)},
                level="ERROR",
            )
            raise

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main logic.

        Args:
            input_data: Input payload from A2A or direct invocation

        Returns:
            Output payload
        """
        pass

    def register_mcp_tool(
        self,
        descriptor: MCPToolDescriptor,
        handler: Callable,
    ):
        """Register an MCP tool for this agent."""
        self.tool_registry.register_tool(descriptor, handler)
