from langgraph.types import interrupt

from agents import (
    orchestrator,
    scientist,
    general_agent,
    executor_agent,
)

MAX_DECISIONS = 8


def get_dialogue(state):
    return "\n\n".join(state["dialogue"])


def add_turn(state, speaker, response):
    return {
        "dialogue": [f"[{speaker}]: {response}"]
    }


def orchestrator_step(state):
    # Stop after 8 Orchestrator decisions.
    if state["decision_count"] >= MAX_DECISIONS:
        return {
            "last_decision": {
                "role": "finish",
                "instruction": "Decision limit reached."
            }
        }

    response = orchestrator(
        get_dialogue(state),
        state["task"]
    )

    # Read the Orchestrator's decision.
    role = "scientist"
    instruction = "Plan the task."
    note = ""
    question, kind, after = "", "ambiguous", "executor"

    for line in response.splitlines():
        line = line.strip().replace("*", "")

        if line.lower().startswith("next role:"):
            role = line.split(":", 1)[1].strip().lower()

        elif line.lower().startswith("instruction:"):
            instruction = line.split(":", 1)[1].strip()

        elif line.lower().startswith("question:"):
            question = line.split(":", 1)[1].strip()

        elif line.lower().startswith("kind:"):
            kind = line.split(":", 1)[1].strip().lower()

        elif line.lower().startswith("after:"):
            after = line.split(":", 1)[1].strip().lower()

    # Standardize role names.
    for name in ["scientist", "general", "executor", "human", "finish"]:
        if name in role:
            role = name
            break

    if role not in ["scientist", "general", "executor", "human", "finish"]:
        role = "scientist"

    # A traceback or non-zero exit code goes to the executor, not general.
    latest = state["dialogue"][-1] if state["dialogue"] else ""
    if (
        role == "general"
        and latest.startswith("[Executor]:")
        and "Exit code:" in latest
        and "Exit code: 0" not in latest
    ):
        role = "executor"
        instruction = (
            "Repair the script that just failed. Read the traceback in the "
            "dialogue and keep the Scientist's plan unchanged."
        )
        note = (
            "\n[Note: the last run failed, so the turn goes to the "
            "executor instead of the general agent.]"
        )

    # Do not finish before a successful run and test RMSE.
    if role == "finish" and not has_test_rmse(state["dialogue"]):
        has_plan = any(
            turn.startswith("[Scientist]:")
            for turn in state["dialogue"]
        )

        if has_plan:
            role = "executor"
            instruction = (
                "Run the Python script and print the training and test RMSE. "
                "Check the shared dialogue and fix any previous errors."
            )
        else:
            role = "scientist"
            instruction = (
                "Create a clear implementation plan for the Silverbox "
                "Neural ODE task before the Executor writes code."
            )

        note = (
            "\n[Note: finish was refused because no successful run has "
            f"printed a test RMSE yet, so the turn goes to the {role}.]"
        )

    # Details of a human stop (used only when role is "human").
    kind = "hard" if "hard" in kind else "ambiguous"
    after = "finish" if "finish" in after else "executor"

    # Nothing can be reported before a successful run printed a test RMSE.
    if role == "human" and kind == "hard" and after == "finish" and not has_test_rmse(state["dialogue"]):
        after = "executor"

    if role == "human" and not question:
        question = "Your request can be read in more than one way. Which reading should I follow?"

    return {
        "dialogue": [f"[Orchestrator]: {response}{note}"],
        "decision_count": state["decision_count"] + 1,
        "last_decision": {
            "role": role,
            "instruction": instruction,
            "question": question,
            "kind": kind,
            "after": after
        }
    }


def human_step(state):
    # Pause the graph (LangGraph interrupt) and wait for the person's reply.
    # The value is shown to the person. When the same thread is resumed,
    # interrupt() returns the reply.
    decision = state["last_decision"]

    reply = interrupt({
        "kind": decision["kind"],
        "question": decision["question"]
    })

    turns = [f"[human]: {reply}"]

    # Ambiguous request: the orchestrator reads the reply and decides again.
    if decision["kind"] == "ambiguous":
        return {
            "dialogue": turns,
            "last_decision": {"role": "orchestrator", "instruction": ""}
        }

    # Hard decision accepted (the reply starts with "y"): the step runs now.
    if reply.strip().lower().startswith("y"):
        return {
            "dialogue": turns,
            "last_decision": {
                "role": decision["after"],
                "instruction": decision["instruction"]
            }
        }

    # Hard decision refused: the step does not run, the turn returns to the orchestrator.
    turns.append(
        f"[System]: The person refused. The step '{decision['after']}: "
        f"{decision['instruction']}' did NOT run. No script was written or run, "
        f"and nothing was overwritten or reported."
    )
    return {
        "dialogue": turns,
        "last_decision": {"role": "orchestrator", "instruction": ""}
    }


def scientist_step(state):
    decision = state["last_decision"]

    response = scientist(
        get_dialogue(state),
        decision["instruction"],
        state["task"]
    )

    return add_turn(state, "Scientist", response)


def general_step(state):
    decision = state["last_decision"]

    response = general_agent(
        get_dialogue(state),
        decision["instruction"],
        state["task"]
    )

    return add_turn(state, "General Agent", response)


def executor_step(state):
    decision = state["last_decision"]

    # The Scientist must provide a plan before execution.
    has_plan = any(
        turn.startswith("[Scientist]:")
        for turn in state["dialogue"]
    )

    if not has_plan:
        response = (
            "Execution not started because the Scientist has not provided "
            "a plan. The Orchestrator should ask the Scientist to plan first."
        )
    else:
        response = executor_agent(
            get_dialogue(state),
            decision["instruction"],
            state["task"]
        )

    return add_turn(state, "Executor", response)


def has_test_rmse(dialogue):
    # Verify successful execution and a printed test RMSE.
    for turn in dialogue:
        if not turn.startswith("[Executor]:"):
            continue

        if "Exit code: 0" not in turn:
            continue

        if "STDOUT:" not in turn:
            continue

        stdout = turn.split("STDOUT:", 1)[1].split(
            "STDERR:", 1
        )[0]

        for line in stdout.splitlines():
            line = line.strip().lower()

            if "test rmse" in line and (
                ":" in line or "=" in line
            ):
                # Confirm there is a numeric value after ':' or '='.
                value = line.replace("=", ":", 1).split(":", 1)[-1].strip()

                try:
                    float(value.split()[0])
                    return True
                except (ValueError, IndexError):
                    continue

    return False