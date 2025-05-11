#!/usr/bin/env python3

import argparse
import datetime
import os
import subprocess
import sys
import threading
import time
import webbrowser
import re
from pathlib import Path

from remote_config import LOCAL_PATH

LOG_ROOT = os.path.join(LOCAL_PATH, 'results', 'train_logs')


def valid_date(s):
    try:
        datetime.datetime.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def get_latest_logdir():
    subdirs = [
        p for p in Path(LOG_ROOT).iterdir()
        if p.is_dir() and valid_date(p.name)
    ]
    if not subdirs:
        return None
    return max(subdirs, key=lambda p: p.name)


def launch_tensorboard_and_open_browser(logdir, timeout=10):
    proc = subprocess.Popen(
        ["tensorboard", "--logdir", str(logdir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    url_pattern = re.compile(r"http://(localhost|127\.0\.0\.1):\d+/")
    url_found = None
    start_time = time.time()

    def read_stdout():
        nonlocal url_found
        for line in proc.stdout:
            print(line, end="")  # stream to terminal
            match = url_pattern.search(line)
            if match:
                url_found = match.group(0)
                break

    thread = threading.Thread(target=read_stdout)
    thread.start()
    thread.join(timeout=timeout)

    if url_found:
        webbrowser.open(url_found)
    else:
        proc.terminate()
        proc.wait()
        raise RuntimeError("TensorBoard failed to start or did not report a valid local URL within timeout.")

    proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("date", nargs="?", help="Date in YYYY-MM-DD format. Defaults to today.")
    parser.add_argument("-l", "--latest", action="store_true", help="Use latest log directory.")
    args = parser.parse_args()

    if args.latest:
        logdir = get_latest_logdir()
        if not logdir:
            print(f"Error: No valid log directories found in {LOG_ROOT}", file=sys.stderr)
            sys.exit(1)
    else:
        date_str = args.date or datetime.date.today().isoformat()
        if not valid_date(date_str):
            print("Error: Invalid date format. Use YYYY-MM-DD.", file=sys.stderr)
            sys.exit(1)
        logdir = os.path.join(LOG_ROOT, date_str)

    if not os.path.isdir(logdir):
        print(f"Error: Log directory does not exist: {logdir}", file=sys.stderr)
        sys.exit(1)

    try:
        launch_tensorboard_and_open_browser(logdir)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
