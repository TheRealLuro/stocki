#!/usr/bin/env python3
"""Cross-platform launcher for the full Stocki stack (Windows + Linux + macOS).

Starts, in order:
  1. Postgres + backend API   (docker compose, port 8000)
  2. ONNX model API           (uvicorn, port 8001)
  3. Frontend dev server      (Vite, port 5173)

Ctrl+C stops the model API and frontend. The backend stack keeps running;
stop it with `docker compose down` (or `docker compose stop`).

Usage:  python start.py
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_WINDOWS = os.name == "nt"

API_PORT = int(os.environ.get("STOCKI_API_PORT", "8000"))
MODEL_PORT = int(os.environ.get("MODEL_PORT", "8001"))
FRONTEND_PORT = int(os.environ.get("FRONTEND_PORT", "5173"))


def port_in_use(port: int) -> bool:
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def pick_free_port(port: int) -> int:
    while port_in_use(port):
        port += 1
    return port


def http_get(url: str, timeout: float = 2.0) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def wait_for_backend(port: int, seconds: int = 60) -> bool:
    for _ in range(seconds):
        if http_get(f"http://localhost:{port}/health") is not None:
            return True
        time.sleep(1)
    return False


def wait_for_model(port: int, seconds: int = 30) -> bool:
    for _ in range(seconds):
        body = http_get(f"http://localhost:{port}/health")
        if body:
            try:
                if json.loads(body).get("model_loaded") is True:
                    return True
            except json.JSONDecodeError:
                pass
        time.sleep(1)
    return False


def find_uvicorn() -> str | None:
    """Prefer the project venv; fall back to uvicorn on PATH."""
    candidates = [
        ROOT / ".venv" / ("Scripts" if IS_WINDOWS else "bin") / ("uvicorn.exe" if IS_WINDOWS else "uvicorn"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return shutil.which("uvicorn")


def npm_cmd() -> str | None:
    # On Windows npm is npm.cmd; shutil.which resolves that via PATHEXT.
    return shutil.which("npm")


def popen_managed(args: list[str], cwd: Path, env: dict) -> subprocess.Popen:
    """Start a child in its own process group so we can stop the whole tree."""
    kwargs: dict = {"cwd": str(cwd), "env": env}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(args, **kwargs)


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            if IS_WINDOWS:
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def _sigterm_to_interrupt(signum, frame):
    raise KeyboardInterrupt


def main() -> int:
    global MODEL_PORT, FRONTEND_PORT

    # Route SIGTERM through KeyboardInterrupt so the finally-block cleanup runs
    # instead of orphaning the model API / frontend. (Windows console Ctrl+C
    # already raises KeyboardInterrupt.)
    if not IS_WINDOWS:
        signal.signal(signal.SIGTERM, _sigterm_to_interrupt)

    if port_in_use(MODEL_PORT):
        new = pick_free_port(MODEL_PORT)
        print(f"!! Port {MODEL_PORT} was already in use; using {new} for the model API instead.")
        MODEL_PORT = new
    if port_in_use(FRONTEND_PORT):
        new = pick_free_port(FRONTEND_PORT)
        print(f"!! Port {FRONTEND_PORT} was already in use; using {new} for the frontend instead.")
        FRONTEND_PORT = new

    # Dev-friendly rate limits (see start.sh); export yourself to override.
    env = dict(os.environ)
    env.setdefault("STOCKI_RATE_LIMIT", "600")
    env.setdefault("STOCKI_DATASET_RATE_LIMIT", "60")
    env["MODEL_PORT"] = str(MODEL_PORT)
    env["FRONTEND_PORT"] = str(FRONTEND_PORT)

    if shutil.which("docker") is None:
        print("!! docker not found -- install Docker Desktop / Engine first.", file=sys.stderr)
        return 1

    children: list[subprocess.Popen] = []
    try:
        print("==> Starting Postgres + backend API (docker compose)")
        subprocess.run(["docker", "compose", "up", "-d"], cwd=ROOT, env=env, check=True)

        print(f"==> Waiting for the backend API on :{API_PORT}")
        if not wait_for_backend(API_PORT):
            print("!! Backend API did not become healthy after 60s -- check 'docker compose logs api'",
                  file=sys.stderr)
            return 1

        uvicorn = find_uvicorn()
        if uvicorn is None:
            venv_py = ".venv\\Scripts\\pip" if IS_WINDOWS else ".venv/bin/pip"
            venv_create = "python -m venv .venv" if IS_WINDOWS else "python3 -m venv .venv"
            print("!! uvicorn not found -- run:", file=sys.stderr)
            print(f"     {venv_create} && {venv_py} install -r model/requirements.txt", file=sys.stderr)
            return 1

        print(f"==> Starting model API on :{MODEL_PORT}")
        children.append(popen_managed(
            [uvicorn, "main:app", "--port", str(MODEL_PORT)], cwd=ROOT / "model", env=env))

        if not wait_for_model(MODEL_PORT):
            print(f"!! Model API is still not healthy on :{MODEL_PORT}", file=sys.stderr)
            return 1

        npm = npm_cmd()
        if npm:
            if not (ROOT / "frontend" / "node_modules").is_dir():
                print("==> Installing frontend dependencies")
                subprocess.run([npm, "install", "--no-audit", "--no-fund"],
                               cwd=ROOT / "frontend", env=env, check=True)
            print(f"==> Starting frontend dev server on :{FRONTEND_PORT}")
            children.append(popen_managed(
                [npm, "run", "dev", "--", "--host", "0.0.0.0", "--port", str(FRONTEND_PORT)],
                cwd=ROOT / "frontend", env=env))
        else:
            print("!! npm not found; skipping frontend startup. Install Node.js/npm to run the dashboard.")

        print()
        print(f"Dashboard:  http://localhost:{FRONTEND_PORT}")
        print(f"Backend:    http://localhost:{API_PORT}/docs")
        print(f"Model API:  http://localhost:{MODEL_PORT}/health")
        print()
        print("Ctrl+C stops the model API and frontend. 'docker compose down' stops the backend stack.")

        # Wait until a child dies or the user hits Ctrl+C.
        while True:
            for proc in children:
                if proc.poll() is not None:
                    return proc.returncode or 0
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n==> Shutting down model API and frontend...")
        return 0
    finally:
        for proc in children:
            stop_process(proc)


if __name__ == "__main__":
    sys.exit(main())
