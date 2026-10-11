import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import write_and_run_python

# Load the API key from .env
load_dotenv()
api_key = os.getenv("API_KEY")

# Connect to Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0
)

# The last script the executor wrote, so a repair can start from it.
last_script = ""


def to_text(content):
    """Convert a LangChain message .content (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


# Note: The 8-decision limit is enforced in nodes.py (orchestrator_step).
def orchestrator(dialogue, task):
    prompt = f"""
You are the orchestrator of a five-destination system.

Task and background:
{task}

Conversation so far:
{dialogue}

Choose the next destination based on the dialogue.

Available destinations:
- scientist: develops the scientific plan
- general: explains concepts, dataset details, or what a reported error (such as RMSE) measures
- executor: writes and runs Python code
- human: stops the system and asks the person a question
- finish: ends the workflow

Rules:
1. Make a model-based decision based on the dialogue.
2. Do not follow a fixed sequence of roles.
3. Do not write theory, scientific explanations, or code yourself.
4. You have no coding tools.
5. Choose finish only if an executor message confirms that a
   Python script actually ran and printed the test RMSE.
6. If the task is not complete, choose the role best suited
   to the next step.
7. Questions about the equation, network, or fitting plan go to the
   scientist.
8. If the latest executor message shows a traceback or a non-zero
   exit code, choose executor so it can repair the script. Do not
   send tracebacks to general.
9. Choose human in exactly two situations.
   AMBIGUOUS: the latest user request has more than one reading, and
   the readings would send the work to different roles or would change
   the number that gets reported. A request is also ambiguous when it
   does not say whether to use results already on disk or to train and
   measure a new model. For example, "How good is the model?" can mean
   explain what the error measures (general), or train and measure a
   new model (executor), so ask which one is wanted. Set Kind to
   ambiguous and ask which reading to follow.
   HARD: the next step would run the coding tool, overwrite a result
   that is already on disk, or end the dialogue and report a
   measurement. Set Kind to hard, state that step clearly in the
   Question, and ask whether to proceed.
   Decide AMBIGUOUS before HARD: if the latest user request has more
   than one reading, ask about the reading first, even if the next step
   would also run the coding tool.
10. For a hard decision, After is the destination that runs if the
   person accepts (executor or finish), and Instruction is the
   instruction that destination will follow.
11. If the person refused a step, that step did not run. Do not
   propose the same step again unchanged. Ask the person what to do
   instead, or propose a changed step as a new hard decision.
12. Do not ask the person about something they already answered in
   the dialogue. If the person accepted a step, it has already run.

Return exactly these lines, each on a separate line. Write the
Question, Kind and After lines only when the next destination is human.
Next role: scientist, general, executor, human, or finish
Instruction: what that destination should do
Question: the question to ask the person
Kind: ambiguous or hard
After: executor or finish
Reason: one sentence explaining your choice

Do not include any other text.
"""
    return to_text(llm.invoke(prompt).content)



def scientist(dialogue, instruction, task):
    prompt = f"""
You are the scientist in a four-role system.

Task:
{task}

Instruction:
{instruction}

Conversation so far:
{dialogue}

Your responsibilities:
1. Write the differential equation for the system.
2. Propose the neural network architecture, including its inputs
   and output.
3. Write the state equations to be integrated.
4. Explain how to train the neural network using the training data.
5. Explain how to integrate the state equations using the
   fourth-order Runge-Kutta method (RK4).
6. Explain how to perform free-run prediction, using the input
   signal but not measured output values to update test predictions.
7. Explain how to calculate training and test RMSE and plot
   predicted versus measured output.
8. Provide numbered implementation steps specific enough for
   the executor to follow.

For the Silverbox Neural ODE task, consider the equation:

y'' + a*y' + b*y + c*y^3 = g*u(t)

A neural model may learn acceleration from the current output,
velocity, and input:

y' = v
v' = alpha_theta(y, v, u)

Numerical advice (the executor follows your plan, so state it in the plan):
- In volts and seconds the numbers differ greatly in size: y is about 0.05,
  the velocity about 20 and the acceleration about 10^4. A network trained
  on such numbers learns badly, and the rollout can collapse to zero.
- Scale the signals. Divide y by its standard deviation on the training
  samples (YS) and u by its standard deviation on the training samples (US).
  Let W0 be the standard deviation of the finite-difference velocity on the
  training samples divided by YS. Divide the velocity by W0*YS and integrate
  with RK4 in the scaled time, with the step H = W0*dt. The network then
  predicts the scaled acceleration, and its inputs and its output are of
  order 1.
- Compute the scaling constants from the training samples only, and convert
  the prediction back to volts before computing the RMSE.
- Train on short windows (for example 20 samples) with Adam and plan enough
  epochs to converge (for example 60). The whole script must finish within
  five minutes on a CPU, so ask the executor to stop training after about
  three minutes.

Rules:
- You are responsible for the scientific theory and implementation plan.
- Do not write Python code.
- You have no coding tools.
- Do not claim a numerical result was measured unless an executor
  message has already reported that result.
- Make the plan clear and detailed enough to implement.

Return the scientific explanation and numbered implementation steps.
"""
    return to_text(llm.invoke(prompt).content)



def general_agent(dialogue, instruction, task):
    prompt = f"""
You are the general agent in a four-role system.

Task:
{task}

Instruction:
{instruction}

Conversation so far:
{dialogue}

Your role is to answer high-level questions related to the task.

You can explain:
- What records or variables in the dataset represent.
- What technical terms and equations mean.
- What a reported error or RMSE measures.

Rules:
- Do not write Python code.
- You have no coding tools.
- Do not create or replace the scientist's fitting plan.
- If the question requires choosing or developing the differential
  equation or fitting method, that is the scientist's responsibility.
- If a traceback needs to be diagnosed or code needs to be fixed,
  that is the executor's responsibility.
- Do not invent results or claim that code has been executed.

Answer the instruction clearly and in simple language.
"""
    return to_text(llm.invoke(prompt).content)



def executor_agent(dialogue, instruction, task):
    global last_script

    prompt = f"""
You are the executor in a four-role system.

Task:
{task}

Orchestrator instruction:
{instruction}

Conversation so far:
{dialogue}

Previous script written by the executor (empty if there is none):
{last_script}

Your job is to implement the latest plan provided by the Scientist.

First, identify the latest Scientist message in the conversation.
Follow that plan for the differential equation, neural network,
data preparation, training, and free-run evaluation.

Requirements for this Silverbox Neural ODE task:
- Load SNLS80mV.mat from the same directory as the generated script.
- Load the file using:
      from pathlib import Path
      from scipy.io import loadmat
      data_path = Path(__file__).parent / "SNLS80mV.mat"
      data = loadmat(data_path)
- Inspect the available MATLAB keys and array shapes to identify
  the correct input and output variables. Do not guess their names.
- Use the excerpt [42650:46746].
- Center the input and output.
- Train on the first 3072 samples.
- Test on the last 1024 samples using free-run prediction.
- Do ONE free run over the whole excerpt: start at the first sample
  with the measured position and a finite-difference velocity, and
  integrate through the training samples into the test samples.
  The rollout may read u but never the measured y.
- The whole script must finish within 5 minutes on a CPU. Train on
  short rollout windows and use the full free run only for evaluation.
- Calculate and print the actual training and test RMSE.
- Print the training RMSE in this format: Train RMSE: <numeric value>
- Print the test RMSE in this format: Test RMSE: <numeric value>
- Plot predicted versus measured output.
- Before the two RMSE lines, print a short summary, one item per line: the
  network (number of layers, hidden units, activation), the optimizer and
  learning rate, the window length and batch size, the number of epochs
  completed, the final training loss and the training time in seconds.
  Also print the training loss about every 10 epochs. These extra lines must
  not contain the word RMSE.

Return only complete Python source code.
Do not use Markdown code fences.
Do not invent results.
Do not call plt.show(); save the figure with plt.savefig instead.
If the latest Executor message in the conversation shows a failure
(non-zero exit code or a traceback), write a corrected version of
the previous script and keep the Scientist's plan unchanged.
"""

    code = to_text(llm.invoke(prompt).content).strip()

    # If Gemini wrote text around a fenced block, keep only the block.
    if "```" in code:
        code = code.split("```")[1]
        if code.startswith("python"):
            code = code[len("python"):]
        code = code.strip()

    # Remove code fences if Gemini includes them.
    if code.startswith("```"):
        lines = code.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        code = "\n".join(lines)

    last_script = code

    # Write and execute the generated Python script.
    result = write_and_run_python(code)

    return (
        f"Executor ran: {result['filename']}\n"
        f"Exit code: {result['exit_code']}\n"
        f"STDOUT:\n{result['stdout']}\n"
        f"STDERR:\n{result['stderr']}"
    )