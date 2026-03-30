from typing import TypedDict, Literal, Annotated
from langgraph.graph.message import add_messages


class SupervisorState(TypedDict):
    """Top-level graph state. Manages the plan and delegates to sub-graphs."""

    messages: Annotated[list, add_messages]
    user_query: str

    # Planning
    plan: list[str]
    current_task_index: int
    task_type: Literal["research", "code", "hybrid", "done"]

    # Results from sub-graphs
    research_results: list[dict]
    code_results: list[dict]

    # Final output
    final_answer: str
    needs_human_approval: bool
    human_feedback: str | None

    # Safety
    iteration_count: int
    max_iterations: int


class RAGState(TypedDict):
    """State for the Corrective RAG sub-graph."""

    messages: Annotated[list, add_messages]
    query: str
    original_query: str

    # Retrieval
    documents: list[dict]
    relevance_scores: list[float]
    relevance_threshold: float

    # CRAG control
    retrieval_grade: Literal["relevant", "ambiguous", "irrelevant"]
    rewrite_count: int
    max_rewrites: int
    use_web_search: bool

    # Output
    generation: str
    citations: list[str]


class CodeState(TypedDict):
    """State for the Code Generation & Verification sub-graph."""

    messages: Annotated[list, add_messages]
    task_description: str
    context: str

    # Code generation
    generated_code: str
    language: Literal["python", "sql", "shell"]

    # Execution
    execution_result: str
    execution_error: str | None
    exit_code: int | None

    # Self-correction
    retry_count: int
    max_retries: int
    critic_feedback: str | None

    # Output
    verified_output: str
    is_verified: bool
