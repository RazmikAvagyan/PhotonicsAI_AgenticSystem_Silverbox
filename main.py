import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END

from nodes import orchestrator_step, scientist_step, general_step, executor_step, has_test_rmse

TASK = """
Complete Task 5: a Neural ODE model for the Silverbox dataset.

Data: SNLS80mV.mat in the working directory. V1 is the input u, V2 is the output y,
fs = 610.35 Hz, dt = 1/fs. Use the excerpt [42650:46746] and subtract the mean of u and of y.
Train on the first 3072 samples, test on the last 1024.

The circuit obeys y'' + a*y' + b*y + c*y^3 = g*u(t). Do not fit a, b, c, g directly.
Model the acceleration with a fully connected PyTorch network:
    y' = v,   v' = alpha_theta(y, v, u)
Use RK4 with step dt, driven by the measured u.

Test with ONE free run over the whole excerpt: start at the first sample with the measured position and
a finite-difference velocity, integrate through the training samples and on into the test samples.
The rollout may read u but never the measured y.

The script must finish within 5 minutes on a CPU, save the test plot with plt.savefig (no plt.show()),
and print these two lines exactly, in volts:
    Train RMSE: <value>
    Test RMSE: <value>

The Scientist plans, the Executor writes and runs the script. The Orchestrator chooses roles from the
dialogue, and finishes only after a successful run printed a test RMSE, or after 8 decisions.
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


if __name__ == "__main__":
    start = {
        "task": TASK,
        "dialogue": ["[User]: " + TASK.strip()],
        "decision_count": 0,
        "last_decision": {"role": "orchestrator", "instruction": ""},
    }

    shown = 0
    state = start
    try:
        for state in app.stream(start, config={"recursion_limit": 60}, stream_mode="values"):
            for turn in state["dialogue"][shown:]:
                print("=" * 70 + "\n" + turn + "\n")
            shown = len(state["dialogue"])
    except Exception as error:
        # For example the API quota ran out: keep what was produced so far.
        print("=" * 70)
        print("Run stopped early:", type(error).__name__, str(error)[:300])

    dialogue = state["dialogue"]
    print("=" * 70)
    print("Orchestrator decisions:", state["decision_count"])
    print("Test RMSE printed by a successful run:", has_test_rmse(dialogue))
    for turn in dialogue:
        if turn.startswith("[Executor]:") and "Exit code: 0" in turn:
            print("\n".join(l for l in turn.splitlines() if "rmse" in l.lower()))

    with open("dialogue_log.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(dialogue))