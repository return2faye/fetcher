"""Docker sandbox: execute code in the fetcher-sandbox container."""

import docker

SANDBOX_CONTAINER = "fetcher-sandbox"
EXEC_TIMEOUT = 30  # seconds


def execute_in_sandbox(
    code: str,
    language: str = "python",
    timeout: int = EXEC_TIMEOUT,
) -> dict:
    """Execute code inside the sandbox container.

    Returns {"stdout": str, "stderr": str, "exit_code": int}.
    """
    client = docker.from_env()

    try:
        container = client.containers.get(SANDBOX_CONTAINER)
    except docker.errors.NotFound:
        return {
            "stdout": "",
            "stderr": f"Sandbox container '{SANDBOX_CONTAINER}' not found. Run ./scripts/start.sh",
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

    try:
        exec_result = container.exec_run(
            cmd,
            user="sandbox",
            workdir="/home/sandbox",
            demux=True,
        )

        stdout = exec_result.output[0].decode("utf-8", errors="replace") if exec_result.output[0] else ""
        stderr = exec_result.output[1].decode("utf-8", errors="replace") if exec_result.output[1] else ""

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exec_result.exit_code,
        }

    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Execution error: {str(e)}",
            "exit_code": 1,
        }
