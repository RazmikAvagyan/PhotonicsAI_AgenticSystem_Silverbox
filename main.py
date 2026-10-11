import operator
import sys
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from nodes import (
    orchestrator_step, scientist_step, general_step, executor_step, human_step, has_test_rmse
)

# The Silverbox facts that all roles see. The request typed by the user is added after them.
TASK = """
Background: the Silverbox work from Homework 1 (a Neural ODE model of the Silverbox oscillator).
Data: SNLS80mV.mat in the working directory. V1 is the input u, V2 is the output y,
fs = 610.35 Hz, dt = 1/fs. Excerpt [42650:46746], mean of u and of y subtracted.
First 3072 samples = training, last 1024 = test.
Model: y' = v, v' = alpha_theta(y, v, u), a fully connected PyTorch network, RK4 with step dt.
Metric: ONE free run over the whole excerpt from sample 0 (measured position and finite-difference
velocity, reads only u). Train RMSE on the 3072 training samples, test RMSE on the 1024 test samples.
Predicting zero gives about 0.06 V on the test samples.
If a script is written, it must finish within 5 minutes on a CPU, save plots with plt.savefig
(no plt.show()), and print these two lines exactly, in volts:
    Train RMSE: <value>
    Test RMSE: <value>
Results of earlier runs are already on disk (generated_scripts/ and the saved plots).
"""


class State(TypedDict):
    task: str
    dialogue: Annotated[list[str], operator.add]  # new turns are appended
    decision_count: int
    last_decision: dict


def choose_next(state):
    role = state["last_decision"]["role"]
    return role if role in ["scientist", "general", "executor", "human"] else END


def after_human(state):
    # human_step stored where the turn goes: back to the orchestrator,
    # or on to the step the person accepted (the executor, or the end).
    role = state["last_decision"]["role"]
    return role if role in ["orchestrator", "executor"] else END


graph = StateGraph(State)
graph.add_node("orchestrator", orchestrator_step)
graph.add_node("scientist", scientist_step)
graph.add_node("general", general_step)
graph.add_node("executor", executor_step)
graph.add_node("human", human_step)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges("orchestrator", choose_next)
graph.add_conditional_edges("human", after_human)
for role in ["scientist", "general", "executor"]:
    graph.add_edge(role, "orchestrator")  # the turn goes back to the orchestrator
# The checkpointer stores the state of the thread, so a paused graph can be resumed.
app = graph.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    # The request comes from the command line, or is typed when the program asks.
    request = " ".join(sys.argv[1:]) or input("Your request: ")

    start = {
        "task": TASK + "\nUser request:\n" + request,
        "dialogue": ["[User]: " + request],
        "decision_count": 0,
        "last_decision": {"role": "orchestrator", "instruction": ""},
    }
    config = {"configurable": {"thread_id": "run-1"}, "recursion_limit": 60}

    shown = 0
    state = start
    graph_input = start
    try:
        while True:
            # Run (or resume) the thread until it finishes or pauses.
            for state in app.stream(graph_input, config, stream_mode="values"):
                for turn in state["dialogue"][shown:]:
                    print("=" * 70 + "\n" + turn + "\n")
                shown = len(state["dialogue"])

            snapshot = app.get_state(config)
            if not snapshot.next:  # nothing left to run: the graph has finished
                break


            # The graph is paused at the human node: show the question, wait for the reply.
            ask = snapshot.tasks[0].interrupts[0].value
            print("#" * 70)
            print(f"GRAPH PAUSED ({ask['kind']} decision). Question:")
            print(ask["question"])
            if ask["kind"] == "hard":
                print("(type yes to accept, anything else refuses)")
            reply = input("Your reply: ")
            graph_input = Command(resume=reply)  # resume the SAME thread with the reply
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