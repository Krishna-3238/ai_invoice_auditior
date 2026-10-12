"""In-process tool dispatch. Pipeline agents call local tool functions directly."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _registry():
    from backned.tools.invoice_watcher_tool import invoice_watcher_tool
    from backned.tools.data_harvester_tool import data_harvester_tool
    from backned.tools.lang_bridge_tool import lang_bridge_tool
    from backned.tools.data_completeness_checker_tool import data_completeness_checker_tool
    from backned.tools.business_validation_tool import business_validation_tool
    from backned.tools.insight_reporter_tool import insight_reporter_tool
    from backned.tools.rag_tools.vector_indexer_tool import vector_indexer_tool
    from backned.tools.rag_tools.semantic_retriever_tool import semantic_retriever_tool
    from backned.tools.rag_tools.chunk_ranker_tool import chunk_ranker_tool
    from backned.tools.rag_tools.response_synthesizer_tool import response_synthesizer_tool
    return {
        "invoice_watcher": invoice_watcher_tool,
        "data_harvester": data_harvester_tool,
        "lang_bridge": lang_bridge_tool,
        "data_completeness_checker": data_completeness_checker_tool,
        "business_validation": business_validation_tool,
        "insight_reporter": insight_reporter_tool,
        "vector_indexer": vector_indexer_tool,
        "semantic_retriever": semantic_retriever_tool,
        "chunk_ranker": chunk_ranker_tool,
        "response_synthesizer": response_synthesizer_tool,
    }


def get_tools_sync(use_cache: bool = True):
    return list(_registry().values())


def get_tool_by_name(tool_name: str, use_cache: bool = True):
    try:
        return _registry()[tool_name]
    except KeyError as exc:
        raise ValueError(f"Unknown local tool: {tool_name}") from exc


def invoke_tool_sync(tool_name: str, tool_input: dict, use_cache: bool = True):
    return get_tool_by_name(tool_name).invoke(tool_input)
