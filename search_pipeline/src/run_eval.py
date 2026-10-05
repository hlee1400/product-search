"""
Runner entrypoint that produces a clean log file from the full pipeline.

Why this exists: when you run `src.main` directly, the output is a mix of
useful prints (eval metrics, sample hits) and noise — warnings, HF/torch
init messages, tqdm progress bars that repaint every second, etc. This
runner suppresses the noise BEFORE any of those libraries are imported and
tees stdout to both the terminal and a timestamped log file.

Usage:
    python -m src.run_eval
"""
from __future__ import annotations

# ─── 1. Noise suppression (must happen before any heavy imports) ──────────────
import os
os.environ.setdefault("TQDM_DISABLE", "1")                       # tqdm progress bars
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")       # huggingface_hub downloads
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")         # transformers logger
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")  # transformers advisories
os.environ.setdefault("PYTHONWARNINGS", "ignore")                # python warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")                    # OpenMP conflict guard

import warnings
warnings.filterwarnings("ignore")

import logging
for name in (
    "transformers", "sentence_transformers", "faiss",
    "torch", "urllib3", "huggingface_hub", "filelock",
):
    logging.getLogger(name).setLevel(logging.ERROR)

# ─── 2. Tee stdout to a log file; drop stderr (warnings end up there) ─────────
import sys
from datetime import datetime
from pathlib import Path


class _Tee:
    """Writes to multiple streams. Drops carriage-return rewrites (tqdm)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        # If any \r remains, keep only the final segment — that's the last
        # frame of a progress bar, not the in-between repaints.
        if "\r" in s:
            s = s.rsplit("\r", 1)[-1]
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()

    # File-like protocol methods that libraries (transformers, click, etc.)
    # probe before deciding whether to colorize / paint progress bars.
    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        # Forward to the first underlying stream that has a real fd, else
        # raise OSError — which is how Python signals "no fd" for fileno().
        for st in self.streams:
            try:
                return st.fileno()
            except (AttributeError, OSError, ValueError):
                continue
        raise OSError("Tee has no underlying fileno")

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        for st in self.streams:
            enc = getattr(st, "encoding", None)
            if enc:
                return enc
        return "utf-8"


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import LOGS_DIR  # noqa: E402  (must follow sys.path setup)

LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

_log_file = open(LOG_PATH, "w", encoding="utf-8")
_real_stdout = sys.__stdout__
_real_stderr = sys.__stderr__
sys.stdout = _Tee(_real_stdout, _log_file)
sys.stderr = open(os.devnull, "w", encoding="utf-8")  # silence warnings / framework prints

# ─── 3. Run main ──────────────────────────────────────────────────────────────
from src.main import main


def _restore_streams() -> None:
    try:
        _log_file.flush()
        _log_file.close()
    finally:
        sys.stdout = _real_stdout
        sys.stderr = _real_stderr


if __name__ == "__main__":
    print(f"writing log to {LOG_PATH}\n")
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc(file=_log_file)
        traceback.print_exc(file=_real_stderr)
        _restore_streams()
        raise
    _restore_streams()
    print(f"done. log saved to {LOG_PATH}")
