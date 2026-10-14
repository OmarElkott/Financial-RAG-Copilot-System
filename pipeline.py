"""Run the complete Finance RAG pipeline from the project root.

This script:
1. Ingests HTML documents.
2. Chunks the documents.
3. Creates embeddings.
4. Rebuilds and indexes Qdrant.
5. Starts the FastAPI server.
6. Sends a sample question through the API.

Usage:
    python pipeline.py

The script expects Qdrant to already be running on localhost:6333.
"""

from pathlib import Path
import json
import os
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
HOST = "127.0.0.1"
PORT = 8000
API_URL = f"http://{HOST}:{PORT}"


def environment() -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + current if current else "")
    return env


def run_module(module: str) -> None:
    print(f"\n>>> Running {module}")
    subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT,
        env=environment(),
        check=True,
    )


def wait_for_api(timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(f"{API_URL}/", timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, ConnectionError):
            time.sleep(1)
    raise RuntimeError("FastAPI did not start within the timeout period.")


def ask_sample_question() -> None:
    payload = json.dumps({
        "query": "By how much did Microsoft Cloud revenue increase in fiscal year 2025?",
        "top_k": 5,
    }).encode("utf-8")
    request = Request(
        f"{API_URL}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"The API returned HTTP {error.code}: {details}") from error

    print("\n>>> Sample API response")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    for module in ("ingest", "chunking", "embed", "index_qdrant"):
        run_module(module)

    print("\n>>> Starting FastAPI")
    server = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "api:app",
            "--host", HOST,
            "--port", str(PORT),
        ],
        cwd=ROOT,
        env=environment(),
    )

    try:
        wait_for_api()
        print(f"API is running at {API_URL}")
        print(f"Interactive docs: {API_URL}/docs")
        ask_sample_question()
        print("\nPipeline completed. Press Ctrl+C to stop the API.")
        server.wait()
    except KeyboardInterrupt:
        print("\nStopping API...")
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()