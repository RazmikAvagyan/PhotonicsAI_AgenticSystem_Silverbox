import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# One working directory per run: generated_scripts/run_<date>_<time>.
# It is created once, when the program starts, so every script of the same
# run lands in the same folder and a new run never mixes with an old one.
RUN_DIR = (
    Path(__file__).parent
    / "generated_scripts"
    / datetime.now().strftime("run_%Y%m%d_%H%M%S")
)


def write_and_run_python(code):
    # Save every script as a NEW numbered file in the working directory of this run.
    folder = RUN_DIR
    folder.mkdir(parents=True, exist_ok=True)

    # The generated scripts load the data from their own folder
    # (Path(__file__).parent), so put a copy of the .mat file there.
    mat_source = Path(__file__).parent / "SNLS80mV.mat"
    mat_copy = folder / "SNLS80mV.mat"
    if mat_source.exists() and not mat_copy.exists():
        shutil.copy(mat_source, mat_copy)

    number = len(list(folder.glob("attempt_*.py"))) + 1
    script_path = folder / f"attempt_{number}.py"
    script_path.write_text(code, encoding="utf-8")

    try:
        # Run from the working directory of this run, so the script finds
        # SNLS80mV.mat and saves its plot next to the script.
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=folder,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "MPLBACKEND": "Agg"}  # never open a plot window
        )

        return {
            "filename": str(script_path),
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.TimeoutExpired:
        return {
            "filename": str(script_path),
            "exit_code": -1,
            "stdout": "",
            "stderr": "Execution timed out after 5 minutes."
        }


# Focusing tool: the executor calls it with the settings it chose.
from bench import run_focusing

# name: (default, smallest, largest)
FOCUS_SETTINGS = {
    "macropixel": (4, 1, 16),         # micromirrors per macropixel side (must be 1, 2, 4, 8 or 16)
    "focus_size": (1, 1, 5),          # side of the focus square, in camera pixels
    "background_width": (4, 1, 8),    # thickness of the background ring, in camera pixels
    "n_regions": (4, 2, 8),           # number of background regions
    "iterations": (1000, 50, 20000),  # number of flip-and-measure steps
    "flips": (1, 1, 8),               # macropixels flipped in one step
    "seed": (0, 0, 10**6),            # random start pattern and noise
}


def run_focusing_tool(settings):
    # Use the default for a missing setting and stay inside the allowed range.
    values = {}
    for name, (default, low, high) in FOCUS_SETTINGS.items():
        values[name] = min(max(int(settings.get(name, default)), low), high)
    if values["macropixel"] not in (1, 2, 4, 8, 16):
        values["macropixel"] = 4

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    number = len(list(RUN_DIR.glob("focus_*.png"))) + 1
    result = run_focusing(save_to=RUN_DIR / f"focus_{number}.png", **values)
    result["settings"] = values
    return result