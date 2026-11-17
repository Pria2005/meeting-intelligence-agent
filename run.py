"""Start backend + frontend together:  python run.py   (Ctrl+C stops both)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    env = {**os.environ, "API_URL": os.getenv("API_URL", "http://127.0.0.1:8000")}
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000",
                            "--reload"], cwd=ROOT / "backend", env=env)
    ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"],
                          cwd=ROOT / "frontend", env=env)
    print("\n  API  → http://127.0.0.1:8000/docs\n  App  → http://localhost:8501\n")
    try:
        while api.poll() is None and ui.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in (api, ui):
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()
