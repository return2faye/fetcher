"""Integration nodes: wire real RAG and Code sub-graphs into the Supervisor.

These nodes translate between SupervisorState and sub-graph states,
invoke the compiled sub-graphs, and merge results back.
"""

from fetcher.state import SupervisorState
from fetcher.graphs.rag import compile_rag, create_rag_initial_state
from fetcher.graphs.code import compile_code, create_code_initial_state
from fetcher.utils.memory import store_result, recall_context

# Compile sub-graphs once at module level (they are reusable)
_rag_app = None
_code_app = None


def _get_rag_app():
    global _rag_app
    if _rag_app is None:
        _rag_app = compile_rag()
    return _rag_app


def _get_code_app():
    global _code_app
    if _code_app is None:
        _code_app = compile_code()
    return _code_app


def _extract_task_description(plan: list[str], index: int) -> str:
    """Extract the task description from a plan entry like '[research] Find info'."""
    if index >= len(plan):
        return "unknown task"
    task = plan[index]
    # Strip the [type] prefix
    for prefix in ("[research] ", "[code] ", "[hybrid] "):
        if task.startswith(prefix):
            return task[len(prefix):]
    return task


def rag_node(state: SupervisorState) -> dict:
    """Invoke the real RAG sub-graph for a research task."""
    idx = state["current_task_index"]
    task_desc = _extract_task_description(state["plan"], idx)

    # Recall any relevant past context from long-term memory
    past_context = recall_context(task_desc)

    # Build query: task description + any relevant past context
    query = task_desc
    if past_context:
        query = f"{task_desc}\n\nRelevant prior knowledge:\n{past_context}"

    try:
        # Invoke RAG sub-graph
        rag_state = create_rag_initial_state(query)
        result = _get_rag_app().invoke(rag_state)

        generation = result.get("generation", "")
        citations = result.get("citations", [])
        used_web = result.get("use_web_search", False)
    except Exception as e:
        generation = f"(RAG sub-graph failed: {e})"
        citations = []
        used_web = False

    research_result = {
        "task": task_desc,
        "answer": generation,
        "citations": citations,
        "used_web_search": used_web,
    }

    # Store in long-term memory for future recall
    store_result(task_desc, generation, result_type="research")

    return {
        "research_results": state.get("research_results", []) + [research_result],
        "current_task_index": idx + 1,
    }


def code_node(state: SupervisorState) -> dict:
    """Invoke the real Code sub-graph for a code task."""
    idx = state["current_task_index"]
    task_desc = _extract_task_description(state["plan"], idx)

    # Gather context from previous research results
    context_parts = []
    for r in state.get("research_results", []):
        context_parts.append(r.get("answer", ""))
    context = "\n\n".join(context_parts)

    try:
        # Invoke Code sub-graph
        code_state = create_code_initial_state(task_desc, context=context)
        result = _get_code_app().invoke(code_state)

        code_result = {
            "task": task_desc,
            "output": result.get("verified_output", result.get("execution_result", "")),
            "code": result.get("generated_code", ""),
            "is_verified": result.get("is_verified", False),
            "retries": result.get("retry_count", 0),
        }
    except Exception as e:
        code_result = {
            "task": task_desc,
            "output": f"(Code sub-graph failed: {e})",
            "code": "",
            "is_verified": False,
            "retries": 0,
        }

    # Store in long-term memory
    output_text = f"Code: {code_result['code']}\nOutput: {code_result['output']}"
    store_result(task_desc, output_text, result_type="code")

    return {
        "code_results": state.get("code_results", []) + [code_result],
        "current_task_index": idx + 1,
    }


def hybrid_node(state: SupervisorState) -> dict:
    """Run RAG first, then Code with research context. Single task, two sub-graphs."""
    idx = state["current_task_index"]
    task_desc = _extract_task_description(state["plan"], idx)

    # --- Step 1: Research ---
    past_context = recall_context(task_desc)
    query = task_desc
    if past_context:
        query = f"{task_desc}\n\nRelevant prior knowledge:\n{past_context}"

    try:
        rag_state = create_rag_initial_state(query)
        rag_result = _get_rag_app().invoke(rag_state)

        generation = rag_result.get("generation", "")
        citations = rag_result.get("citations", [])
        used_web = rag_result.get("use_web_search", False)
    except Exception as e:
        generation = f"(RAG sub-graph failed: {e})"
        citations = []
        used_web = False

    research_result = {
        "task": task_desc,
        "answer": generation,
        "citations": citations,
        "used_web_search": used_web,
    }

    store_result(task_desc, generation, result_type="research")

    # --- Step 2: Code (with research context) ---
    all_research = state.get("research_results", []) + [research_result]
    context = "\n\n".join(r.get("answer", "") for r in all_research)

    try:
        code_state = create_code_initial_state(task_desc, context=context)
        code_result_raw = _get_code_app().invoke(code_state)

        code_result = {
            "task": task_desc,
            "output": code_result_raw.get("verified_output", code_result_raw.get("execution_result", "")),
            "code": code_result_raw.get("generated_code", ""),
            "is_verified": code_result_raw.get("is_verified", False),
            "retries": code_result_raw.get("retry_count", 0),
        }
    except Exception as e:
        code_result = {
            "task": task_desc,
            "output": f"(Code sub-graph failed: {e})",
            "code": "",
            "is_verified": False,
            "retries": 0,
        }

    output_text = f"Code: {code_result['code']}\nOutput: {code_result['output']}"
    store_result(task_desc, output_text, result_type="code")

    return {
        "research_results": all_research,
        "code_results": state.get("code_results", []) + [code_result],
        "current_task_index": idx + 1,
    }
