"""
NeuroCamo Tactical Software Launcher
====================================
Launches the NeuroCamo SNN Ground Control Station & Analysis Suite
as a standalone native desktop application window.
"""

import os
import subprocess
import sys
import time
import urllib.request
import webbrowser

HOST = "127.0.0.1"
PORT = 8080
URL = f"http://{HOST}:{PORT}"

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = os.path.join(HERE, "neurocamo", "venv", "Scripts", "python.exe")
APP_PY = os.path.join(HERE, "neurocamo-web", "app.py")


def is_server_running():
    try:
        with urllib.request.urlopen(f"{URL}/api/status", timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_server():
    if is_server_running():
        print(f"  [OK] Backend server already running at {URL}")
        return None

    print(f"  [1/2] Starting NeuroCamo Backend Server at {URL}...")
    proc = subprocess.Popen(
        [PYTHON_EXE, APP_PY],
        cwd=os.path.join(HERE, "neurocamo-web"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Wait for server to come up
    for _ in range(30):
        time.sleep(0.5)
        if is_server_running():
            print("  [OK] Backend server is online and healthy!")
            return proc

    print("  [WARN] Server took longer than expected to report status.")
    return proc


def launch_desktop_window():
    print(f"  [2/2] Opening NeuroCamo GCS Desktop Application Window...")

    # Edge App Mode Path
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]

    for p in edge_paths + chrome_paths:
        if os.path.exists(p):
            subprocess.Popen(
                [
                    p,
                    f"--app={URL}",
                    "--window-size=1520,960",
                    "--window-position=50,50",
                    "--disable-extensions",
                    "--disable-plugins",
                ]
            )
            print(f"  [OK] Launched native application shell using: {os.path.basename(p)}")
            return

    # Fallback to default web browser
    webbrowser.open(URL)
    print("  [OK] Opened in default web browser.")


def main():
    print("=" * 68)
    print("  NEUROCAMO: TACTICAL GROUND CONTROL & SNN SUITE")
    print("=" * 68)
    srv = start_server()
    launch_desktop_window()
    print("=" * 68)
    print("  Software is active. Press Ctrl+C in this window to stop.")
    print("=" * 68)

    if srv:
        try:
            srv.wait()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            srv.terminate()


if __name__ == "__main__":
    main()
