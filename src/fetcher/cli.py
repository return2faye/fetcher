"""CLI runner for Fetcher — interactive query interface with HITL and streaming."""

import asyncio
import sys
import uuid

from fetcher.config import LANGSMITH_TRACING_ENABLED
from fetcher.graphs.supervisor import build_supervisor_graph
from fetcher.state import SupervisorState

from langgraph.checkpoint.sqlite import SqliteSaver
from fetcher.config import SQLITE_DB_PATH


def _print_header():
    print("\n" + "=" * 60)
    print("  Fetcher — Multi-Agent Research & Code System")
    print("=" * 60)
    if LANGSMITH_TRACING_ENABLED:
        print("  [LangSmith tracing: ON]")
    print()


def _print_plan(plan: list[str]):
    print("\n--- Execution Plan ---")
    for i, task in enumerate(plan, 1):
        print(f"  {i}. {task}")
    print()


def _get_human_feedback(final_answer: str) -> str:
    """Prompt the user for HITL feedback on the synthesized answer."""
    print("\n--- Final Answer (pending approval) ---")
    print(final_answer)
    print("\n--- Human Review ---")
    print("Options:")
    print("  [Enter]          → Approve")
    print("  reject:<reason>  → Reject and re-plan")
    print("  <your feedback>  → Revise with instructions")
    feedback = input("\nYour feedback: ").strip()
    return feedback


async def run_streaming(query: str):
    """Run the supervisor graph with streaming output and HITL interrupts."""
    graph = build_supervisor_graph(use_stubs=False)
    checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{SQLITE_DB_PATH}")
    app = graph.compile(checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_query": query,
        "messages": [],
    }

    print(f"\nProcessing: {query}\n")

    while True:
        # Stream events from the graph
        current_node = None
        async for event in app.astream_events(initial_state, config=config, version="v2"):
            kind = event.get("event", "")

            # Track which node is executing
            if kind == "on_chain_start" and event.get("name"):
                node_name = event["name"]
                if node_name not in ("LangGraph", "__start__") and ":" not in node_name:
                    if node_name != current_node:
                        current_node = node_name
                        print(f"\n  [{current_node}] ", end="", flush=True)

            # Stream LLM tokens
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    print(chunk.content, end="", flush=True)

        print()  # Newline after streaming

        # Check the current state for interrupt
        state = app.get_state(config)

        if state.next:
            # Graph is interrupted — needs human input
            snapshot_values = state.values
            final_answer = snapshot_values.get("final_answer", "")

            if snapshot_values.get("plan"):
                _print_plan(snapshot_values["plan"])

            feedback = _get_human_feedback(final_answer)

            # Resume with the human's feedback using Command
            from langgraph.types import Command
            initial_state = Command(resume=feedback)
        else:
            # Graph completed
            final_state = state.values
            print("\n" + "=" * 60)
            print("FINAL ANSWER:")
            print("=" * 60)
            print(final_state.get("final_answer", "(no answer)"))
            print()
            break


def run_sync(query: str):
    """Run the supervisor graph synchronously (no streaming) with HITL."""
    graph = build_supervisor_graph(use_stubs=False)
    checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{SQLITE_DB_PATH}")
    app = graph.compile(checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_input = {
        "user_query": query,
        "messages": [],
    }

    print(f"\nProcessing: {query}\n")

    while True:
        result = app.invoke(initial_input, config=config)

        # Check if graph is interrupted
        state = app.get_state(config)

        if state.next:
            final_answer = result.get("final_answer", "")

            if result.get("plan"):
                _print_plan(result["plan"])

            feedback = _get_human_feedback(final_answer)

            from langgraph.types import Command
            initial_input = Command(resume=feedback)
        else:
            print("\n" + "=" * 60)
            print("FINAL ANSWER:")
            print("=" * 60)
            print(result.get("final_answer", "(no answer)"))
            print()
            break


def main():
    _print_header()

    # Parse args
    streaming = "--stream" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        query = " ".join(args)
    else:
        query = input("Enter your query: ").strip()
        if not query:
            print("No query provided. Exiting.")
            return

    if streaming:
        asyncio.run(run_streaming(query))
    else:
        run_sync(query)


if __name__ == "__main__":
    main()
