import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from tools import run_focusing_tool, FOCUS_SETTINGS

# Load the API key from .env
load_dotenv()
api_key = os.getenv("API_KEY")

# Connect to Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0
)

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


# Note: The decision limit is enforced in nodes.py (orchestrator_step).
def orchestrator(dialogue, task):
    prompt = f"""
You are the orchestrator of a four-role system.

Task:
{task}

Conversation so far:
{dialogue}

Choose the next role based on the dialogue.

Available roles:
- scientist: plans the focusing experiments and explains the physics
- general: explains concepts, such as what the enhancement measures
- executor: calls the focusing tool for ONE experiment with the settings you give it
- finish: ends the workflow

Rules:
1. Make a model-based decision based on the dialogue.
2. Do not follow a fixed sequence of roles.
3. Do not write theory, scientific explanations, or code yourself.
4. You have no tools.
5. Choose finish only if executor messages report the enhancement of at
   least four experiments with different settings, and you can name the
   best settings and the best enhancement.
6. Each executor call is one experiment. In the Instruction, give the
   executor the settings to try (macropixel, focus_size, background_width,
   n_regions, iterations, flips). Choose them from the scientist's plan and
   from the results so far. Do not repeat settings that were already run.
7. The executor cannot run before the scientist has written a plan, so
   when there is no scientist message yet, choose the scientist first.
   Questions about the physics or the plan go to the scientist.
8. If the latest executor message says the tool could not be called, choose
   executor again with corrected settings.

Return exactly these three fields, each on a separate line:
Next role: scientist, general, executor, or finish
Instruction: what that role should do
Reason: one sentence explaining your choice

Do not include any other text.
"""
    return to_text(llm.invoke(prompt).content)



def scientist(dialogue, instruction, task):
    ranges = "\n".join(
        f"- {name}: {low} to {high}" for name, (default, low, high) in FOCUS_SETTINGS.items()
    )
    prompt = f"""
You are the scientist in a four-role system.

Task:
{task}

Instruction:
{instruction}

Conversation so far:
{dialogue}

Your responsibilities:
1. Explain in a few lines how the camera image is formed
   (E_out = |T E_in|^2, with the DMD changing only the amplitude) and why a
   random pattern gives speckle and a good pattern gives a focus.
2. Define the enhancement as the focus intensity divided by the mean
   intensity of the background regions.
3. Explain what each setting changes and what it costs:
   macropixel size (fewer, larger macropixels converge faster but reach a
   lower enhancement; many small ones need many more iterations),
   focus size (the focus is a focus_size x focus_size square of camera pixels,
   and a speckle grain is about one pixel, so a larger square averages bright
   and dark grains and lowers the enhancement), the ideal enhancement grows
   roughly in proportion to the number of macropixels, which is why many small
   macropixels reach the highest enhancement but need about 20 or more
   iterations per macropixel, background width and number of regions,
   number of iterations,
   number of macropixels flipped per step, and the noise of the camera.
4. Plan exactly 4 experiments as a numbered list. Each line gives the settings
   and the reason for trying them, so that the results can be compared.
   Cover the macropixel size with a large, a medium and a small value, and
   let the fourth experiment compare one other setting (focus size or
   flips) against the best of the first three. Give the small-macropixel
   experiment enough iterations (about 20 per macropixel or more).
   Keep the settings inside the ranges the tool accepts:
{ranges}

Rules:
- You are responsible for the physics and the experiment plan.
- Do not write Python code.
- You have no tools.
- Do not claim a numerical result was measured unless an executor
  message has already reported that result.
- If the instruction asks you to analyse results, use only the numbers
  reported in the conversation and propose the next experiments.
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
- What the DMD, the scattering medium, the camera and the speckle are.
- What the enhancement and the settings of the tool mean.
- What a reported result means.

Rules:
- Do not write Python code.
- You have no tools.
- Do not create or replace the scientist's experiment plan.
- Do not invent results or claim that an experiment has been run.

Answer the instruction clearly and in simple language.
"""
    return to_text(llm.invoke(prompt).content)



def executor_agent(dialogue, instruction, task):
    names = ", ".join(FOCUS_SETTINGS)
    prompt = f"""
You are the executor in a four-role system. Your only tool is the focusing
tool, which runs one search on the simulated bench.

Task:
{task}

Orchestrator instruction:
{instruction}

Conversation so far:
{dialogue}

Choose the settings for ONE experiment. Follow the orchestrator instruction;
where it leaves a setting open, use the latest scientist plan.
The settings are: {names}.

Return only one JSON object with these settings as integers, for example:
{{"macropixel": 4, "focus_size": 1, "background_width": 4, "n_regions": 4, "iterations": 1000, "flips": 1, "seed": 0}}
No other text and no Markdown code fences. Do not invent results.
"""
    text = to_text(llm.invoke(prompt).content).strip()

    # Keep only the JSON object, in case the model wrote text around it.
    start, end = text.find("{"), text.rfind("}")
    try:
        settings = json.loads(text[start:end + 1])
        result = run_focusing_tool(settings)
    except Exception as error:
        return f"Executor could not call the tool: {type(error).__name__}: {error}\nModel output: {text[:300]}"

    return (
        "Executor called run_focusing\n"
        f"Settings: {json.dumps(result['settings'])}\n"
        f"Enhancement: {result['enhancement']:.2f}\n"
        f"Start enhancement (random pattern): {result['start_enhancement']:.2f}\n"
        f"Macropixels: {result['macropixels']}\n"
        f"Camera images: {result['image']}"
    )