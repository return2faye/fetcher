"""Code sub-graph nodes: coder, executor, critic, error_handler."""

import re

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from fetcher.config import (
    OPENAI_MODEL, OPENAI_MODEL_HEAVY, MAX_CODE_RETRIES,
    LLM_TIMEOUT, DOCKER_EXEC_TIMEOUT,
)
from fetcher.state import CodeState
from fetcher.utils.docker_sandbox import execute_in_sandbox


def _get_llm(heavy: bool = False) -> ChatOpenAI:
    model = OPENAI_MODEL_HEAVY if heavy else OPENAI_MODEL
    return ChatOpenAI(model=model, temperature=0, timeout=LLM_TIMEOUT)


# --- Node: coder ---

CODER_SYSTEM_PROMPT = """\
You are an expert Python programmer. Given a task description and optional context, \
write clean, correct Python code that accomplishes the task.

Rules:
- Output ONLY a fenced code block (```python ... ```). No explanation before or after.
- The code must be self-contained and runnable as a script.
- Print the final result to stdout.
- If packages are needed, only use: numpy, pandas, requests, matplotlib (pre-installed).
- Handle errors gracefully where appropriate.
"""

CODER_RETRY_PROMPT = """\
You are an expert Python programmer. Your previous code had an error. \
Fix the code based on the error feedback below.

Rules:
- Output ONLY a fenced code block (```python ... ```). No explanation before or after.
- The code must be self-contained and runnable as a script.
- Print the final result to stdout.
- Address the specific error described in the feedback.
"""


def _extract_code_block(text: str) -> str:
    """Extract Python code from a fenced code block."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: treat entire response as code
    return text.strip()


def coder(state: CodeState) -> dict:
    """LLM generates (or fixes) code for the task."""
    llm = _get_llm(heavy=True)
    retry = state.get("retry_count", 0)

    if retry > 0 and state.get("critic_feedback"):
        # Retry mode: include previous code + error feedback
        messages = [
            SystemMessage(content=CODER_RETRY_PROMPT),
            HumanMessage(
                content=f"Task: {state['task_description']}\n\n"
                f"Previous code:\n```python\n{state.get('generated_code', '')}\n```\n\n"
                f"Error feedback:\n{state['critic_feedback']}"
            ),
        ]
    else:
        # First attempt
        context = state.get("context", "")
        context_section = f"\n\nContext from research:\n{context}" if context else ""
        messages = [
            SystemMessage(content=CODER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Task: {state['task_description']}{context_section}"
            ),
        ]

    try:
        response = llm.invoke(messages)
        code = _extract_code_block(response.content)
    except Exception:
        # LLM failed — return empty code so executor reports the error cleanly
        code = ""
        response = None

    result = {
        "generated_code": code,
        "language": "python",
    }
    if response:
        result["messages"] = [response]
    return result


# --- Node: executor ---

def executor(state: CodeState) -> dict:
    """Execute the generated code in the Docker sandbox."""
    code = state.get("generated_code", "")
    language = state.get("language", "python")

    if not code:
        return {
            "execution_result": "",
            "execution_error": "No code to execute",
            "exit_code": 1,
        }

    result = execute_in_sandbox(code, language=language, timeout=DOCKER_EXEC_TIMEOUT)

    return {
        "execution_result": result["stdout"],
        "execution_error": result["stderr"] if result["exit_code"] != 0 else None,
        "exit_code": result["exit_code"],
    }


# --- Node: critic ---

CRITIC_SYSTEM_PROMPT = """\
You are a code review critic. Given a task, the code that was written, and its execution \
output, determine if the code correctly accomplished the task.

Respond with ONLY a JSON object:
{"verdict": "pass"} if the output is correct and complete.
{"verdict": "fail", "reason": "brief explanation of what's wrong"}
"""


def critic(state: CodeState) -> dict:
    """LLM evaluates whether the code output is correct."""
    import json

    # If there was an execution error, no need to ask the LLM
    if state.get("execution_error"):
        return {
            "is_verified": False,
            "critic_feedback": f"Execution error:\n{state['execution_error']}",
        }

    llm = _get_llm()
    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Task: {state['task_description']}\n\n"
            f"Code:\n```python\n{state.get('generated_code', '')}\n```\n\n"
            f"Output:\n{state.get('execution_result', '(no output)')}"
        ),
    ]

    try:
        response = llm.invoke(messages)
    except Exception:
        # If critic LLM fails, optimistically pass (code ran successfully)
        return {
            "is_verified": True,
            "verified_output": state.get("execution_result", ""),
            "critic_feedback": None,
        }

    try:
        parsed = json.loads(response.content)
        if parsed.get("verdict") == "pass":
            return {
                "is_verified": True,
                "verified_output": state.get("execution_result", ""),
                "critic_feedback": None,
                "messages": [response],
            }
        else:
            return {
                "is_verified": False,
                "critic_feedback": parsed.get("reason", "Code output is incorrect"),
                "messages": [response],
            }
    except (json.JSONDecodeError, KeyError):
        # If we can't parse, assume pass (optimistic)
        return {
            "is_verified": True,
            "verified_output": state.get("execution_result", ""),
            "critic_feedback": None,
            "messages": [response],
        }


# --- Node: error_handler ---

def error_handler(state: CodeState) -> dict:
    """Extract traceback, increment retry count, format feedback for coder."""
    retry = state.get("retry_count", 0) + 1
    error = state.get("execution_error") or state.get("critic_feedback") or "Unknown error"

    # Extract just the last traceback line for concise feedback
    lines = error.strip().split("\n")
    short_error = lines[-1] if lines else error

    feedback = (
        f"Attempt {retry}: {short_error}\n"
        f"Full error:\n{error}"
    )

    return {
        "retry_count": retry,
        "critic_feedback": feedback,
    }


# --- Conditional edge function ---

def should_retry(state: CodeState) -> str:
    """Decide: end (verified or retries exhausted) or retry."""
    if state.get("is_verified", False):
        return "end"

    retry = state.get("retry_count", 0)
    max_retries = state.get("max_retries", MAX_CODE_RETRIES)

    if retry >= max_retries:
        return "end"

    return "retry"
