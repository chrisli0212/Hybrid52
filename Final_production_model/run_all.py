#!/usr/bin/env python3
"""
Launch both prediction service and dashboard.

Usage:
    python run_all.py
    python run_all.py --port 8050
    python run_all.py --no-predictions  # Dashboard only
    python run_all.py --predictions-only  # Prediction service only
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Launch Hybrid51 prediction service and Dash dashboard"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Dashboard port (default: 8050)",
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Skip launching prediction_service.py (dashboard only)",
    )
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Run prediction service only, no dashboard",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Dash in debug mode",
    )
    return parser.parse_args()


def _start_prediction_service():
    """Start prediction_service.py as a subprocess."""
    script = SCRIPT_DIR / "prediction_service.py"
    if not script.exists():
        print(f"[run_all] ERROR: {script} not found")
        sys.exit(1)

    print(f"[run_all] Starting prediction service: {script}")
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    return proc


def _stream_output(proc, prefix="[prediction]"):
    """Non-blocking read of subprocess stdout, print with prefix."""
    import select

    if proc.stdout and proc.stdout.fileno():
        ready, _, _ = select.select([proc.stdout], [], [], 0.1)
        if ready:
            line = proc.stdout.readline()
            if line:
                print(f"{prefix} {line.rstrip()}")


def _run_dashboard(port, debug):
    """Import and run the Dash dashboard in the main process."""
    dashboard_path = SCRIPT_DIR / "theta_dashboard_v4_modern.py"
    if not dashboard_path.exists():
        print(f"[run_all] ERROR: {dashboard_path} not found")
        sys.exit(1)

    print(f"[run_all] Starting dashboard on port {port}")

    # Import the dashboard module
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "theta_dashboard_v4_modern", str(dashboard_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Run the Dash app
    mod.app.run(host="0.0.0.0", port=port, debug=debug)


def main():
    args = _parse_args()

    prediction_proc = None
    shutdown_requested = False

    def _handle_signal(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            # Second signal — force exit
            print("\n[run_all] Forced exit.")
            sys.exit(1)
        shutdown_requested = True
        print("\n[run_all] Shutting down...")
        if prediction_proc and prediction_proc.poll() is None:
            prediction_proc.terminate()
            try:
                prediction_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                prediction_proc.kill()
            print("[run_all] Prediction service stopped.")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # --- Start prediction service ---
    if not args.no_predictions:
        prediction_proc = _start_prediction_service()
        # Give it a moment to initialize
        time.sleep(2)
        if prediction_proc.poll() is not None:
            print(
                f"[run_all] ERROR: Prediction service exited with code "
                f"{prediction_proc.returncode}"
            )
            if not args.predictions_only:
                print("[run_all] Continuing with dashboard only...")
            else:
                sys.exit(1)
        else:
            print("[run_all] Prediction service running (PID: "
                  f"{prediction_proc.pid})")

    # --- Start dashboard or wait ---
    if args.predictions_only:
        print("[run_all] Running prediction service only. Ctrl+C to stop.")
        try:
            while not shutdown_requested:
                if prediction_proc and prediction_proc.poll() is not None:
                    print(
                        f"[run_all] Prediction service exited with code "
                        f"{prediction_proc.returncode}"
                    )
                    break
                _stream_output(prediction_proc)
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    else:
        try:
            _run_dashboard(args.port, args.debug)
        except KeyboardInterrupt:
            pass

    # --- Cleanup ---
    if prediction_proc and prediction_proc.poll() is None:
        print("[run_all] Stopping prediction service...")
        prediction_proc.terminate()
        try:
            prediction_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            prediction_proc.kill()

    print("[run_all] All services stopped.")


if __name__ == "__main__":
    main()
