"""Supervisor graph nodes: intake_planner, router, synthesizer, human_review, finalize."""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from fetcher.config import OPENAI_MODEL, OPENAI_MODEL_HEAVY, MAX_PLAN_ITERATIONS
from fetcher.state import SupervisorState

PLANNER_SYSTEM_PROMPT = """\
You are a task planner. Given a user query, decompose it into a list of sequential sub-tasks.

Each sub-task must be ONE of these types:
- "research": requires searching/retrieving information
- "code": requires writing and executing code
- "hybrid": requires research first, then code

Respond with ONLY a valid JSON object in this exact format (no markdown, no extra text):
{"tasks": [{"description": "...", "type": "research|code|hybrid"}, ...]}

Rules:
- Keep tasks focused and atomic.
- Order them logically (dependencies first).
- Typically 1-4 tasks. Do not over-decompose.
"""


def get_llm(heavy: bool = False) -> ChatOpenAI:
    model = OPENAI_MODEL_HEAVY if heavy else OPENAI_MODEL
    return ChatOpenAI(model=model, temperature=0)


def intake_planner(state: SupervisorState) -> dict:
    """Decompose the user query into a plan of typed sub-tasks."""
    import json

    llm = get_llm()
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=state["user_query"]),
    ]
    response = llm.invoke(messages)

    try:
        parsed = json.loads(response.content)
        tasks = parsed["tasks"]
    except (json.JSONDecodeError, KeyError):
        # Fallback: treat entire query as a single research task
        tasks = [{"description": state["user_query"], "type": "research"}]

    plan = [f"[{t['type']}] {t['description']}" for t in tasks]
    first_type = tasks[0]["type"] if tasks else "done"

    return {
        "messages": [response],
        "plan": plan,
        "current_task_index": 0,
        "task_type": first_type,
        "research_results": [],
        "code_results": [],
        "iteration_count": 0,
        "max_iterations": MAX_PLAN_ITERATIONS,
        "needs_human_approval": False,
        "human_feedback": None,
        "final_answer": "",
    }


def router(state: SupervisorState) -> dict:
    """Advance to the next task in the plan, or signal done."""
    idx = state["current_task_index"]
    plan = state["plan"]
    iteration = state.get("iteration_count", 0) + 1

    if idx >= len(plan) or iteration > state.get("max_iterations", MAX_PLAN_ITERATIONS):
        return {"task_type": "done", "iteration_count": iteration}

    current_task = plan[idx]

    if current_task.startswith("[research]"):
        task_type = "research"
    elif current_task.startswith("[code]"):
        task_type = "code"
    elif current_task.startswith("[hybrid]"):
        task_type = "hybrid"
    else:
        task_type = "research"

    return {"task_type": task_type, "iteration_count": iteration}


def route_by_task_type(state: SupervisorState) -> str:
    """Conditional edge function: route based on task_type."""
    return state["task_type"]


# --- Stub nodes for sub-graphs (replaced in Phase 3 & 4) ---

def rag_subgraph_stub(state: SupervisorState) -> dict:
    """Placeholder for RAG sub-graph. Returns a mock research result."""
    idx = state["current_task_index"]
    task = state["plan"][idx] if idx < len(state["plan"]) else "unknown task"

    result = {"task": task, "answer": f"[STUB] Research result for: {task}"}
    return {
        "research_results": state.get("research_results", []) + [result],
        "current_task_index": idx + 1,
    }


def code_subgraph_stub(state: SupervisorState) -> dict:
    """Placeholder for Code sub-graph. Returns a mock code result."""
    idx = state["current_task_index"]
    task = state["plan"][idx] if idx < len(state["plan"]) else "unknown task"

    result = {"task": task, "output": f"[STUB] Code output for: {task}"}
    return {
        "code_results": state.get("code_results", []) + [result],
        "current_task_index": idx + 1,
    }


def hybrid_stub(state: SupervisorState) -> dict:
    """Placeholder for hybrid tasks: research then code."""
    idx = state["current_task_index"]
    task = state["plan"][idx] if idx < len(state["plan"]) else "unknown task"

    research = {"task": task, "answer": f"[STUB] Research for hybrid: {task}"}
    code = {"task": task, "output": f"[STUB] Code for hybrid: {task}"}
    return {
        "research_results": state.get("research_results", []) + [research],
        "code_results": state.get("code_results", []) + [code],
        "current_task_index": idx + 1,
    }


SYNTHESIZER_SYSTEM_PROMPT = """\
You are a synthesis agent. Given the user's original query and the results from research \
and code execution tasks, produce a clear, comprehensive final answer.

Be concise but thorough. Cite sources where available.
"""


def synthesizer(state: SupervisorState) -> dict:
    """Merge all sub-results into a final answer."""
    llm = get_llm(heavy=True)

    context_parts = []
    for r in state.get("research_results", []):
        context_parts.append(f"Research: {r.get('answer', '')}")
    for c in state.get("code_results", []):
        context_parts.append(f"Code output: {c.get('output', '')}")

    context = "\n\n".join(context_parts) or "No sub-task results available."

    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Original query: {state['user_query']}\n\n"
            f"Sub-task results:\n{context}"
        ),
    ]
    response = llm.invoke(messages)

    return {
        "messages": [response],
        "final_answer": response.content,
        "needs_human_approval": True,
    }


def human_review(state: SupervisorState) -> dict:
    """HITL node. In Phase 6 this will use interrupt_before."""
    return {}


def finalize(state: SupervisorState) -> dict:
    """Terminal node — emit the final answer."""
    return {"final_answer": state.get("final_answer", "")}
