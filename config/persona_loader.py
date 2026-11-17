"""
Persona Loader Utility
======================
Utility module for loading agent personas and prompt templates from YAML configuration.

This module provides a centralized way to load prompts dynamically,
making the system more configurable without code changes.
"""

import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PersonaLoader")

DEFAULT_PERSONA_PATH = "./config/persona_invoice_agent.yaml"

_persona_cache: Optional[Dict[str, Any]] = None


def load_personas(config_path: str = DEFAULT_PERSONA_PATH) -> Dict[str, Any]:
    """
    Load all agent personas from YAML configuration.

    Args:
        config_path: Path to persona YAML file

    Returns:
        Dictionary containing all agent personas and prompts
    """
    global _persona_cache

    if _persona_cache is not None:
        return _persona_cache

    try:
        path = Path(config_path)
        if not path.is_absolute():
            if not path.exists():
                base_dir = Path(__file__).parent.parent
                path = base_dir / config_path

        if not path.exists():
            logger.warning(
                f"Persona config not found at {config_path}. Using empty config."
            )
            return {}

        with open(path, "r", encoding="utf-8") as f:
            _persona_cache = yaml.safe_load(f)

        logger.info(f"Loaded personas from {path}")
        return _persona_cache

    except Exception as e:
        logger.error(f"Failed to load personas from {config_path}: {e}")
        return {}


def get_agent_config(
    agent_name: str, config_path: str = DEFAULT_PERSONA_PATH
) -> Dict[str, Any]:
    """
    Get configuration for a specific agent.

    Args:
        agent_name: Name of the agent (e.g., 'translation_agent', 'validation_agent')
        config_path: Path to persona YAML file

    Returns:
        Agent configuration dictionary
    """
    personas = load_personas(config_path)
    return personas.get(agent_name, {})


def get_prompt(
    agent_name: str, prompt_name: str, config_path: str = DEFAULT_PERSONA_PATH, **kwargs
) -> str:
    """
    Get a specific prompt template for an agent, optionally formatted with kwargs.

    Args:
        agent_name: Name of the agent (e.g., 'translation_agent')
        prompt_name: Name of the prompt (e.g., 'translate_text')
        config_path: Path to persona YAML file
        **kwargs: Variables to format into the prompt template

    Returns:
        Prompt string (formatted if kwargs provided)
    """
    agent_config = get_agent_config(agent_name, config_path)
    prompts = agent_config.get("prompts", {})
    prompt_template = prompts.get(prompt_name, "")

    if not prompt_template:
        logger.warning(f"Prompt '{prompt_name}' not found for agent '{agent_name}'")
        return ""

    if kwargs:
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing key {e} in prompt template '{prompt_name}'")
            return prompt_template

    return prompt_template


def get_agent_persona(
    agent_name: str, config_path: str = DEFAULT_PERSONA_PATH
) -> Dict[str, str]:
    """
    Get persona information for an agent.

    Args:
        agent_name: Name of the agent
        config_path: Path to persona YAML file

    Returns:
        Dictionary with name, persona, role, description
    """
    agent_config = get_agent_config(agent_name, config_path)
    return {
        "name": agent_config.get("name", agent_name),
        "persona": agent_config.get("persona", "Unknown"),
        "role": agent_config.get("role", "No role defined"),
        "description": agent_config.get("description", ""),
    }


def reload_personas(config_path: str = DEFAULT_PERSONA_PATH) -> Dict[str, Any]:
    """
    Force reload personas from file (clears cache).

    Args:
        config_path: Path to persona YAML file

    Returns:
        Freshly loaded personas
    """
    global _persona_cache
    _persona_cache = None
    return load_personas(config_path)


def list_agents(config_path: str = DEFAULT_PERSONA_PATH) -> list:
    """
    List all configured agents.

    Args:
        config_path: Path to persona YAML file

    Returns:
        List of agent names
    """
    personas = load_personas(config_path)
    return list(personas.keys())


def list_prompts(agent_name: str, config_path: str = DEFAULT_PERSONA_PATH) -> list:
    """
    List all prompts for an agent.

    Args:
        agent_name: Name of the agent
        config_path: Path to persona YAML file

    Returns:
        List of prompt names
    """
    agent_config = get_agent_config(agent_name, config_path)
    prompts = agent_config.get("prompts", {})
    return list(prompts.keys())


class AgentPersona:
    """
    Convenience class for accessing agent-specific prompts and configuration.

    Usage:
        persona = AgentPersona('translation_agent')
        prompt = persona.get_prompt('translate_text', source_lang='Spanish', text='Hola')
    """

    def __init__(self, agent_name: str, config_path: str = DEFAULT_PERSONA_PATH):
        """
        Initialize AgentPersona for a specific agent.

        Args:
            agent_name: Name of the agent
            config_path: Path to persona YAML file
        """
        self.agent_name = agent_name
        self.config_path = config_path
        self._config = get_agent_config(agent_name, config_path)

    @property
    def name(self) -> str:
        return self._config.get("name", self.agent_name)

    @property
    def persona(self) -> str:
        return self._config.get("persona", "Unknown")

    @property
    def role(self) -> str:
        return self._config.get("role", "")

    @property
    def description(self) -> str:
        return self._config.get("description", "")

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Get a formatted prompt.

        Args:
            prompt_name: Name of the prompt
            **kwargs: Variables to format into the prompt

        Returns:
            Formatted prompt string
        """
        return get_prompt(self.agent_name, prompt_name, self.config_path, **kwargs)

    def list_prompts(self) -> list:
        """List all available prompts for this agent."""
        return list_prompts(self.agent_name, self.config_path)

    def __repr__(self) -> str:
        return f"AgentPersona(name='{self.name}', persona='{self.persona}')"
