"""
Kaggle submission entry point.

ARC Prize is a CODE competition: you submit a notebook that Kaggle runs
on its own machines with no internet access, and it must write
/kaggle/working/submission.json during the run. The private test set is
swapped in at scoring time, so nothing here may assume the contents of
today's arc-agi_test_challenges.json -- it reads whatever the mounted
dataset provides.

To use, either
  (a) upload this repo as a Kaggle Dataset and paste the cell below into
      a notebook, or
  (b) paste the whole src/ tree into notebook cells.
Option (a) is easier to keep in sync with the repo.

    import sys
    sys.path.insert(0, "/kaggle/input/<your-dataset-name>/src/layers")
    sys.path.insert(0, "/kaggle/input/<your-dataset-name>/src/primitives")
    exec(open("/kaggle/input/<your-dataset-name>/kaggle_notebook.py").read())

Run this file directly for a local dry run of the exact same path.
"""

import os
import sys
import time
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.is_dir():
    for _p in (_SRC_DIR / "layers", _SRC_DIR / "primitives"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

from task_loader import load_all_tasks, get_task_ids, DATA_PATH
from solver import build_kaggle_submission, save_submission


def main():
    started = time.time()
    print(f"data directory: {DATA_PATH}")

    tasks = load_all_tasks("test")
    task_ids = get_task_ids("test")
    print(f"test tasks: {len(task_ids)}")

    submission = build_kaggle_submission(tasks, task_ids, verbose=True)

    # Kaggle requires an entry for EVERY task id in the challenges file.
    # A task that raised or produced nothing still needs a well-formed
    # placeholder, or the whole submission is rejected rather than just
    # scoring zero on that task.
    missing = [t for t in task_ids if t not in submission]
    for task_id in missing:
        submission[task_id] = [
            {"attempt_1": [[0]], "attempt_2": [[0]]}
            for _ in tasks[task_id]["test"]
        ]
    if missing:
        print(f"filled {len(missing)} missing task ids with placeholders")

    save_submission(submission)
    print(f"total runtime: {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
