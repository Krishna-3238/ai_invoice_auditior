"""
Agent-to-Agent (A2A) Protocol
==============================
Standardized discovery and invocation protocol enabling communication
between LangGraph agents and Google ADK agents.

The A2A protocol provides:
1. Agent Cards - Self-describing agent manifests for discovery
2. Message Envelope - Standardized request/response format
3. Agent Registry - Central registry for agent lookup
4. Protocol Bridge - Translates between LangGraph state and ADK message formats
"""

import json
import uuid
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("A2AProtocol")




class AgentFramework(str, Enum):
    """Supported agent frameworks."""

    LANGGRAPH = "langgraph"
    GOOGLE_ADK = "google_adk"


class MessageType(str, Enum):
    """A2A message types."""

    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    DISCOVERY = "discovery"
    HEARTBEAT = "heartbeat"


class AgentStatus(str, Enum):
    """Agent status indicators."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    ERROR = "error"




@dataclass
class AgentCard:
    """
    Agent Card for A2A discovery.

    Each agent publishes a card describing its capabilities,
    accepted input/output formats, and framework type.
    """

    agent_id: str
    name: str
    description: str
    framework: AgentFramework
    version: str = "1.0.0"
    capabilities: List[str] = field(default_factory=list)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    tools: List[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent card to dictionary."""
        d = asdict(self)
        d["framework"] = self.framework.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        """Deserialize agent card from dictionary."""
        data["framework"] = AgentFramework(data["framework"])
        data["status"] = AgentStatus(data.get("status", "active"))
        return cls(**data)




@dataclass
class A2AMessage:
    """
    Standardized message envelope for A2A communication.

    All inter-agent messages are wrapped in this envelope
    regardless of the source/target framework.
    """

    message_id: str
    message_type: MessageType
    source_agent: str
    target_agent: str
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @classmethod
    def create_request(
        cls,
        source: str,
        target: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> "A2AMessage":
        """Create a request message."""
        return cls(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            message_type=MessageType.REQUEST,
            source_agent=source,
            target_agent=target,
            payload=payload,
            correlation_id=correlation_id or f"corr_{uuid.uuid4().hex[:8]}",
        )

    @classmethod
    def create_response(
        cls,
        request: "A2AMessage",
        payload: Dict[str, Any],
    ) -> "A2AMessage":
        """Create a response message for a given request."""
        return cls(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            message_type=MessageType.RESPONSE,
            source_agent=request.target_agent,
            target_agent=request.source_agent,
            payload=payload,
            correlation_id=request.correlation_id,
        )

    @classmethod
    def create_error(
        cls,
        request: "A2AMessage",
        error: str,
    ) -> "A2AMessage":
        """Create an error message for a given request."""
        return cls(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            message_type=MessageType.ERROR,
            source_agent=request.target_agent,
            target_agent=request.source_agent,
            payload={},
            correlation_id=request.correlation_id,
            error=error,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
            "error": self.error,
        }




class AgentRegistry:
    """
    Central registry for agent discovery.

    Agents register their cards, and other agents can discover
    them by ID, framework, or capability.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents = {}                        
            cls._instance._handlers = {}                       
        return cls._instance

    def register(self, card: AgentCard, handler: Optional[Callable] = None):
        """Register an agent with its card and optional handler."""
        self._agents[card.agent_id] = card
        if handler:
            self._handlers[card.agent_id] = handler
        logger.info(
            f"A2A Registry: Registered '{card.name}' "
            f"({card.framework.value}) as '{card.agent_id}'"
        )

    def deregister(self, agent_id: str):
        """Remove an agent from the registry."""
        self._agents.pop(agent_id, None)
        self._handlers.pop(agent_id, None)
        logger.info(f"A2A Registry: Deregistered '{agent_id}'")

    def discover(self, agent_id: str) -> Optional[AgentCard]:
        """Discover an agent by ID."""
        return self._agents.get(agent_id)

    def discover_by_framework(self, framework: AgentFramework) -> List[AgentCard]:
        """Discover all agents of a specific framework."""
        return [card for card in self._agents.values() if card.framework == framework]

    def discover_by_capability(self, capability: str) -> List[AgentCard]:
        """Discover agents with a specific capability."""
        return [
            card for card in self._agents.values() if capability in card.capabilities
        ]

    def get_handler(self, agent_id: str) -> Optional[Callable]:
        """Get the handler function for an agent."""
        return self._handlers.get(agent_id)

    def list_all(self) -> List[AgentCard]:
        """List all registered agents."""
        return list(self._agents.values())

    def reset(self):
        """Reset the registry (for testing)."""
        self._agents.clear()
        self._handlers.clear()




class A2AProtocolBridge:
    """
    Bridge for translating between LangGraph state dicts and
    Google ADK message formats via standardized A2A messages.

    This enables seamless communication between agents running
    on different frameworks.
    """

    def __init__(self):
        self.registry = AgentRegistry()

    def invoke_agent(
        self,
        source_agent_id: str,
        target_agent_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Invoke a target agent from a source agent using A2A protocol.

        Args:
            source_agent_id: ID of the calling agent
            target_agent_id: ID of the target agent
            payload: Data to send to the target agent

        Returns:
            Response payload from the target agent
        """
        request = A2AMessage.create_request(
            source=source_agent_id,
            target=target_agent_id,
            payload=payload,
        )

        logger.info(
            f"A2A: {source_agent_id} → {target_agent_id} " f"[{request.message_id}]"
        )

        remote_env = {
            "adk_extractor_agent": "A2A_EXTRACTOR_URL",
            "adk_translation_agent": "A2A_TRANSLATOR_URL",
            "adk_reporting_agent": "A2A_REPORTER_URL",
        }.get(target_agent_id)
        remote_url = os.getenv(remote_env, "").rstrip("/") if remote_env else ""
        if remote_url:
            import requests
            response = requests.post(
                f"{remote_url}/invoke", json=request.to_dict(),
                timeout=float(os.getenv("A2A_REQUEST_TIMEOUT_SECONDS", "300")),
            )
            response.raise_for_status()
            result = response.json()
            if result.get("message_type") == MessageType.ERROR.value:
                raise RuntimeError(result.get("error", "Remote agent invocation failed"))
            return result.get("payload", result)

        target_card = self.registry.discover(target_agent_id)
        if not target_card:
            raise ValueError(f"Agent '{target_agent_id}' not found locally or via {remote_env}")

        handler = self.registry.get_handler(target_agent_id)
        if not handler:
            raise ValueError(f"No handler registered for agent '{target_agent_id}'")

        try:
            result = handler(request.payload)

            response = A2AMessage.create_response(request, result)

            logger.info(
                f"A2A: {target_agent_id} → {source_agent_id} "
                f"[{response.message_id}] (response to {request.message_id})"
            )

            return response.payload

        except Exception as e:
            error_msg = A2AMessage.create_error(request, str(e))
            logger.error(
                f"A2A ERROR: {target_agent_id} failed: {e} " f"[{error_msg.message_id}]"
            )
            raise

    def langgraph_to_a2a(
        self, state: Dict[str, Any], field_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Convert LangGraph state to A2A payload using field mapping.

        Args:
            state: LangGraph workflow state
            field_mapping: Mapping from state fields to payload fields

        Returns:
            A2A-compatible payload
        """
        payload = {}
        for state_key, payload_key in field_mapping.items():
            if state_key in state:
                payload[payload_key] = state[state_key]
        return payload

    def a2a_to_langgraph(
        self, payload: Dict[str, Any], field_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Convert A2A response payload back to LangGraph state updates.

        Args:
            payload: A2A response payload
            field_mapping: Mapping from payload fields to state fields

        Returns:
            LangGraph state update dictionary
        """
        state_update = {}
        for payload_key, state_key in field_mapping.items():
            if payload_key in payload:
                state_update[state_key] = payload[payload_key]
        return state_update


_registry = AgentRegistry()
_bridge = A2AProtocolBridge()


def get_registry() -> AgentRegistry:
    """Get the global agent registry."""
    return _registry


def get_bridge() -> A2AProtocolBridge:
    """Get the global A2A protocol bridge."""
    return _bridge
