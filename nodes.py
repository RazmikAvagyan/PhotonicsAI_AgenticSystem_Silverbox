from agents import (
    orchestrator,
    scientist,
    general_agent,
    executor_agent,
)

MAX_DECISIONS = 12
MIN_EXPERIMENTS = 4


def get_dialogue(state):
    return "\n\n".join(state["dialogue"])


def add_turn(state, speaker, response):
    return {
        "dialogue": [f"[{speaker}]: {response}"]
    }


def orchestrator_step(state):
    # Stop after MAX_DECISIONS Orchestrator decisions.
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

    for line in response.splitlines():
        line = line.strip().replace("*", "")

        if line.lower().startswith("next role:"):
            role = line.split(":", 1)[1].strip().lower()

        elif line.lower().startswith("instruction:"):
            instruction = line.split(":", 1)[1].strip()

    # Standardize role names.
    for name in ["scientist", "general", "executor", "finish"]:
        if name in role:
            role = name
            break

    if role not in ["scientist", "general", "executor", "finish"]:
        role = "scientist"

    # A failed tool call goes to the executor, not to the general agent.
    latest = state["dialogue"][-1] if state["dialogue"] else ""
    if role == "general" and latest.startswith("[Executor]: Executor could not call"):
        role = "executor"
        instruction = "Call the tool again with corrected settings as JSON integers."
        note = (
            "\n[Note: the last tool call failed, so the turn goes to the "
            "executor instead of the general agent.]"
        )

    # Do not finish before enough experiments have returned an enhancement.
    if role == "finish" and count_experiments(state["dialogue"]) < MIN_EXPERIMENTS:
        has_plan = any(
            turn.startswith("[Scientist]:")
            for turn in state["dialogue"]
        )

        if has_plan:
            role = "executor"
            instruction = (
                "Run another experiment with settings that were not tried yet "
                "and report its enhancement."
            )
        else:
            role = "scientist"
            instruction = (
                "Plan the focusing experiments before the Executor calls the tool."
            )

        note = (
            f"\n[Note: finish was refused because fewer than {MIN_EXPERIMENTS} "
            f"experiments have returned an enhancement, so the turn goes to the {role}.]"
        )

    return {
        "dialogue": [f"[Orchestrator]: {response}{note}"],
        "decision_count": state["decision_count"] + 1,
        "last_decision": {
            "role": role,
            "instruction": instruction
        }
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


def count_experiments(dialogue):
    # Number of executor turns where the tool returned a numeric enhancement.
    count = 0
    for turn in dialogue:
        if not turn.startswith("[Executor]: Executor called run_focusing"):
            continue

        for line in turn.splitlines():
            if line.startswith("Enhancement:"):
                try:
                    float(line.split(":", 1)[1])
                    count += 1
                except ValueError:
                    pass

    return count