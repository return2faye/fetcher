"""CLI runner for Fetcher — interactive query interface with HITL and streaming."""

import argparse
import asyncio
import sqlite3
import sys
import uuid

from fetcher.config import LANGSMITH_TRACING_ENABLED, MAX_QUERY_LENGTH, SQLITE_DB_PATH
from fetcher.graphs.supervisor import build_supervisor_graph

from langgraph.checkpoint.sqlite import SqliteSaver


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
    print("  [Enter]          -> Approve")
    print("  reject:<reason>  -> Reject and re-plan")
    print("  <your feedback>  -> Revise with instructions")
    try:
        feedback = input("\nYour feedback: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nApproved (non-interactive).")
        feedback = "approve"
    return feedback


def _print_final(answer: str):
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(answer or "(no answer)")
    print()


async def run_streaming(query: str):
    """Run the supervisor graph with streaming output and HITL interrupts."""
    graph = build_supervisor_graph(use_stubs=False)
    checkpointer = SqliteSaver(conn=sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False))
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
        try:
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
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            return
        except Exception as e:
            print(f"\n\nError during execution: {e}")
            return

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
            _print_final(final_state.get("final_answer", ""))
            break


def run_sync(query: str):
    """Run the supervisor graph synchronously (no streaming) with HITL."""
    graph = build_supervisor_graph(use_stubs=False)
    checkpointer = SqliteSaver(conn=sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False))
    app = graph.compile(checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_input = {
        "user_query": query,
        "messages": [],
    }

    print(f"\nProcessing: {query}\n")

    while True:
        try:
            result = app.invoke(initial_input, config=config)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            return
        except Exception as e:
            print(f"\nError during execution: {e}")
            return

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
            _print_final(result.get("final_answer", ""))
            break


def main():
    parser = argparse.ArgumentParser(
        prog="fetcher",
        description="Fetcher — Multi-agent research & code verification system",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="The query to process. If omitted, prompts interactively.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable token-level streaming output",
    )
    args = parser.parse_args()

    _print_header()

    if args.query:
        query = " ".join(args.query)
    else:
        try:
            query = input("Enter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if not query:
            print("No query provided. Exiting.")
            return

    if len(query) > MAX_QUERY_LENGTH:
        print(f"Query truncated to {MAX_QUERY_LENGTH} characters.")
        query = query[:MAX_QUERY_LENGTH]

    if args.stream:
        asyncio.run(run_streaming(query))
    else:
        run_sync(query)


if __name__ == "__main__":
    main()
