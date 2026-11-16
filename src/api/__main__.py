"""Start the demo site.

python -m src.api                 # http://127.0.0.1:8000, this laptop only
python -m src.api --lan           # also reachable on the local network (replay only there)
python -m src.api --port 8010
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.api.app import ApiConfig, create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The AML Auditor demo site.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lan", action="store_true", help="listen on the local network too")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    app = create_app(ApiConfig(outputs=args.outputs, lan=args.lan))
    print(f"AML Auditor demo site: http://127.0.0.1:{args.port}")
    uvicorn.run(app, host=host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
