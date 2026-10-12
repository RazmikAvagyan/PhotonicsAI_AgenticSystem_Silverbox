import json
import operator
import re
import sys
from pathlib import Path
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END

import tools
from nodes import orchestrator_step, scientist_step, general_step, executor_step, count_experiments

TASK = """
Focus a beam through a simulated bench: a laser, a digital micromirror device (DMD), a scattering
medium and a camera. The DMD changes only the amplitude of the input field. With T the transmission
matrix of the path, the camera records |T Ein|^2 plus noise, so a second shot of the same pattern is
not a copy of the first.

The goal is a focus: a bright spot in a chosen region of the camera, standing out from the speckle
around it. The focusing tool runs this search:
  - divide the DMD into macropixels and start from a random pattern;
  - after each transmission, read the intensity in the focus region and in the proximal background;
  - split the background into several regions; the enhancement is the focus against that background;
  - flip some macropixels, see how the enhancement changed (keep the flip only if it helped), repeat.

The tool takes these settings (integers): macropixel, focus_size, background_width, n_regions,
iterations, flips, seed. It returns the enhancement and saves the camera images.

The Scientist plans the experiments, the Executor is the only role that calls the tool, and the
Orchestrator chooses roles from the dialogue. The settings are chosen by the agent, not by the user.
The Orchestrator finishes only after at least 4 experiments with different settings have returned an
enhancement, or after 12 decisions.
"""


class State(TypedDict):
    task: str
    dialogue: Annotated[list[str], operator.add]  # new turns are appended
    decision_count: int
    last_decision: dict


def choose_next(state):
    role = state["last_decision"]["role"]
    return role if role in ["scientist", "general", "executor"] else END


graph = StateGraph(State)
graph.add_node("orchestrator", orchestrator_step)
graph.add_node("scientist", scientist_step)
graph.add_node("general", general_step)
graph.add_node("executor", executor_step)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges("orchestrator", choose_next)
for role in ["scientist", "general", "executor"]:
    graph.add_edge(role, "orchestrator")  # the turn goes back to the orchestrator
app = graph.compile()


def load_dialogue(path):
    """Read the turns of a stopped run from its dialogue log, to continue that run."""
    text = Path(path).read_text(encoding="utf-8").strip()
    turns = re.split(r"\n\n(?=\[(?:User|Orchestrator|Scientist|General Agent|Executor)\]: )", text)
    if turns[-1].startswith("[Orchestrator]"):
        turns.pop()                                  # a decision that was never carried out: decide again
    folders = re.findall(r"Camera images: (.+)[/\\]focus_\d+\.png", text)
    if folders:
        tools.RUN_DIR = Path(folders[-1])            # keep the new images in the folder of the first part
    return turns


if __name__ == "__main__":
    # The only thing the user types: the request. The agent chooses all settings.
    # "python main.py --resume" continues the run saved in dialogue_log.txt (for example after a quota stop).
    resume = "--resume" in sys.argv
    request = " ".join(a for a in sys.argv[1:] if a != "--resume") or "Focus the beam."
    turns = load_dialogue("dialogue_log.txt") if resume else ["[User]: " + request]
    start = {
        "task": TASK,
        "dialogue": turns,
        "decision_count": sum(t.startswith("[Orchestrator]") for t in turns),
        "last_decision": {"role": "orchestrator", "instruction": ""},
    }

    shown = len(turns) if resume else 0              # print only the new turns when resuming
    state = start
    try:
        for state in app.stream(start, config={"recursion_limit": 80}, stream_mode="values"):
            for turn in state["dialogue"][shown:]:
                print("=" * 70 + "\n" + turn + "\n")
            shown = len(state["dialogue"])
    except Exception as error:
        # For example the API quota ran out: keep what was produced so far.
        print("=" * 70)
        print("Run stopped early:", type(error).__name__, str(error)[:1500])

    dialogue = state["dialogue"]

    # Table of the experiments the agent ran: settings of each run and the enhancement.
    rows = ["| Run | macropixel | focus_size | background_width | n_regions | iterations | flips | seed | Enhancement |",
            "|---|---|---|---|---|---|---|---|---|"]
    for turn in dialogue:
        if turn.startswith("[Executor]: Executor called run_focusing"):
            lines = {l.split(":", 1)[0]: l.split(":", 1)[1].strip() for l in turn.splitlines() if ":" in l}
            s = json.loads(lines["Settings"])
            rows.append(f"| {len(rows) - 1} | " + " | ".join(str(s[k]) for k in
                        ["macropixel", "focus_size", "background_width", "n_regions", "iterations", "flips", "seed"])
                        + f" | {lines['Enhancement']} |")
    table = "\n".join(rows)

    print("=" * 70)
    print("Orchestrator decisions:", state["decision_count"])
    print("Experiments with an enhancement:", count_experiments(dialogue))
    print(table)

    with open("dialogue_log.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(dialogue))
    with open("experiments_table.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")