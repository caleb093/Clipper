"""Start the local web UI and open it in the browser."""

import socket
import threading
import webbrowser

from clipper.config import load_env

HOST, PORT = "127.0.0.1", 8765


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    with socket.socket() as s:
        if s.connect_ex((HOST, PORT)) == 0:
            print(f"Clipper is already running. Opening {url}")
            webbrowser.open(url)
            return

    load_env()
    import uvicorn

    from clipper.web.app import app

    print(f"Clipper is running at {url}")
    print("Keep this window open while you use it. Press Ctrl+C (or close the window) to stop.")
    threading.Timer(1.5, webbrowser.open, args=[url]).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
