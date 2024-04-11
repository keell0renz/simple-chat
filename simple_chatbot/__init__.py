"""Launch helpers for Poetry."""

import subprocess
import sys

def main():
    """Helper."""
    try:
        subprocess.run(["chainlit", "run", "./simple_chatbot/app.py"], check=True)
    except KeyboardInterrupt:
        sys.exit(0)
