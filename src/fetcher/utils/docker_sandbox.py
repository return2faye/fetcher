"""Docker sandbox: execute code in the fetcher-sandbox container."""

import docker
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

SANDBOX_CONTAINER = "fetcher-sandbox"
EXEC_TIMEOUT = 30  # seconds


def execute_in_sandbox(
    code: str,
    language: str = "python",
    timeout: int = EXEC_TIMEOUT,
) -> dict:
    """Execute code inside the sandbox container.

    Returns {"stdout": str, "stderr": str, "exit_code": int}.
    Enforces a timeout — returns exit_code 124 on timeout (matching `timeout` command convention).
    """
    try:
        client = docker.from_env()
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Docker not available: {e}",
            "exit_code": 1,
        }

    try:
        container = client.containers.get(SANDBOX_CONTAINER)
    except docker.errors.NotFound:
        return {
            "stdout": "",
            "stderr": f"Sandbox container '{SANDBOX_CONTAINER}' not found. Run ./scripts/start.sh",
            "exit_code": 1,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Docker error: {e}",
            "exit_code": 1,
        }

    if container.status != "running":
        return {
            "stdout": "",
            "stderr": f"Sandbox container is not running (status: {container.status})",
            "exit_code": 1,
        }

    if language == "python":
        cmd = ["python", "-c", code]
    elif language == "shell":
        cmd = ["bash", "-c", code]
    else:
        return {
            "stdout": "",
            "stderr": f"Unsupported language: {language}",
            "exit_code": 1,
        }

    def _run():
        return container.exec_run(
            cmd,
            user="sandbox",
            workdir="/home/sandbox",
            demux=True,
        )

    try:
        # Docker SDK exec_run has no timeout param, so we use a thread with timeout
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            exec_result = future.result(timeout=timeout)

        stdout = exec_result.output[0].decode("utf-8", errors="replace") if exec_result.output[0] else ""
        stderr = exec_result.output[1].decode("utf-8", errors="replace") if exec_result.output[1] else ""

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exec_result.exit_code,
        }

    except FuturesTimeoutError:
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
            "exit_code": 124,
        }
    except docker.errors.DockerException as e:
        return {
            "stdout": "",
            "stderr": f"Docker error: {e}",
            "exit_code": 1,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Execution error: {e}",
            "exit_code": 1,
        }
