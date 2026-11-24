"""Start the local web app. Use --open-browser to open it when ready."""

import argparse
import threading
import time
import urllib.error
import urllib.request
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def _open_browser_when_ready(url: str, timeout: float = 60.0) -> None:
    """Open *url* after the local server responds, without a helper console."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                webbrowser.open(url)
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AEGS web application")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="open the web interface when the server is ready",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    url = f"http://{HOST}:{PORT}"
    if args.open_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(url,),
            daemon=True,
            name="aegs-browser-launcher",
        ).start()
    print(f"AEGS starting at {url}")
    uvicorn.run("exam_src.api.main:app", host=HOST, port=PORT, reload=True)
