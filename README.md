# Task 6: Agentic System for the Silverbox Neural-ODE Task

## Overview

This project is a system of four roles built with LangGraph. It is given Task 5 (a Neural-ODE model of the Silverbox oscillator) and tries to solve it: a scientist plans, an executor writes and runs the script, and an orchestrator decides who speaks next.

The four roles are the nodes of one `StateGraph`. They share one state, the dialogue, in which every turn is stored as `[Role]: text`. After a specialist node runs, an edge returns the turn to the orchestrator, so the next speaker sees the new message.

```
START -> orchestrator --(model decision)--> scientist | general | executor | END
              ^                                  |          |          |
              +----------------------------------+----------+----------+
```

## Roles

- **Orchestrator** - reads the dialogue and returns three fields: `Next role`, `Instruction` and `Reason`. The decision is a call to a Gemini model, not a keyword table or a fixed order. It writes no theory and no code and has no tool. It stops on `finish` or after 8 decisions, and it may finish only after an executor turn shows a script that ran (exit code 0) and printed a numeric `Test RMSE`.
- **Scientist** - writes the differential equation, the scaled state equations, the network, the training, the RK4 step, the free-run test, the RMSE and plot, and numbered implementation steps. Its prompt contains numerical advice: the signals span several orders of magnitude (y about 0.05, velocity about 20, acceleration about 10^4), so they should be scaled by their training standard deviations and integrated in a scaled time. It has no tool and does not claim a measured number unless the executor printed it.
- **General agent** - answers high-level questions (what a record is, what a term means, what an error measures). It has no tool and does not write the fitting plan.
- **Executor** - the only role with the coding tool. It writes a complete Python script for the latest scientist plan, runs it, and returns the result or the traceback. The script prints a short summary of the run (network, optimizer, epochs, loss, time) before the two RMSE lines. After a failure the executor writes a corrected script, starting from its previous script, with the same plan.

## Coding tool

`write_and_run_python` in `tools.py` does two things:

1. Writes the script as a new numbered file, `attempt_N.py`, into `generated_scripts/run_<date>_<time>/`. This working directory is created once per run, and a copy of `SNLS80mV.mat` is placed in it.
2. Runs the file with a time limit of 5 minutes and returns the file name, the exit code, the standard output and the standard error. A timeout is reported with exit code `-1`.

## Files

- `main.py` - builds the graph, gives it the Task 5 text, prints the dialogue turn by turn, and writes `dialogue_log.txt` (also if the run stops early, for example when the API quota runs out).
- `agents.py` - the four prompts and the Gemini model; the executor writes the script and calls the coding tool.
- `nodes.py` - the graph nodes: reads the orchestrator decision, enforces the 8-decision limit and the finish rule, and checks the printed test RMSE.
- `tools.py` - the coding tool.
- `dialogue_log.txt` - the full dialogue of the validation run.
- `generated_scripts/run_20261010_151626/` - the script written by the executor in the validation run (`attempt_1.py`) and the plot it saved (`silverbox_neural_ode_free_run.png`).
- `SNLS80mV.mat` - the Silverbox dataset (Wigren and Schoukens, https://www.nonlinearbenchmark.org/). It must be in the project folder.

## Requirements

Python 3.10, with:

```bash
pip install langgraph langchain-google-genai python-dotenv numpy scipy matplotlib torch
```

The generated scripts use NumPy, SciPy, Matplotlib and PyTorch, so install them in the same environment. With a PyTorch version built for NumPy 1.x, use `numpy<2`.

## How to run

1. Create a file named `.env` in the project folder with your Gemini key (do not commit this file):

   ```
   API_KEY=your_key_here
   ```

2. Put `SNLS80mV.mat` in the project folder.
3. Run:

   ```bash
   python main.py
   ```

A run needs about 5 to 16 Gemini requests. The free tier of the Gemini API allows only about 20 requests per model and day, so avoid test runs before the real one.

## Validation on Task 5

The system was given the text of Task 5. The run took 3 of the 8 allowed decisions:

| Turn | Agent | What it did |
|---|---|---|
| 1 | Orchestrator | Chose the scientist, because the model, training and evaluation had to be specified before any code. |
| 2 | Scientist | Wrote the plan: scaled state equations, MLP 3-64-64-1 with Tanh, RK4, training on short windows with Adam, one free run, RMSE and plot. |
| 3 | Orchestrator | Chose the executor. |
| 4 | Executor | Wrote `attempt_1.py` and ran it: exit code 0. |
| 5 | Orchestrator | Chose finish, because the run had printed a test RMSE. |

Printed by the run (free run over the whole excerpt, in volts):

```
Epoch 10 / 60 - Loss: 0.030344
Epoch 20 / 60 - Loss: 0.030818
Epoch 30 / 60 - Loss: 0.027990
Epoch 40 / 60 - Loss: 0.027835
Epoch 50 / 60 - Loss: 0.027449
Epoch 60 / 60 - Loss: 0.026963
Network: 2 hidden layers, 64 units each, Tanh activation
Optimizer: Adam, learning rate = 0.002
Trajectory window length: 24, batch size: 64
Epochs completed: 60
Final training loss: 0.026963
Training time: 105.41 seconds
Train RMSE: 0.012576
Test RMSE: 0.019973
```

## Result

| Quantity | Value |
|---|---:|
| Free-run training RMSE (3072 samples) | 0.012576 V |
| Free-run test RMSE (1024 samples) | 0.019973 V |
| Zero-prediction training RMSE | 0.0530 V |
| Zero-prediction test RMSE | 0.0605 V |

The test RMSE is about 3.0 times lower than the zero-prediction baseline, and 1.59 times the training RMSE. The prediction follows the measured output over the whole excerpt without drifting; the largest errors (up to about 0.10 V near 5.9 s) occur where the output amplitude is largest. The plot is `generated_scripts/run_20261010_151626/silverbox_neural_ode_free_run.png`.

## Notes

- The executor kept the scientist's model, scaling, RK4 step and free-run test, but changed some training details: learning rate 0.002 instead of 0.001, no learning-rate schedule, gradient-norm clipping at 1.0, and a 210 s time guard instead of 180 s.
- The result depends on the scaling advice in the scientist prompt; it is part of the design of the scientist role.
- This is one run with one seed. The training loss was still falling slowly at epoch 60 and the script used 105 s of the 300 s limit, so longer training might lower the error.
- The orchestrator finishes as soon as a script has printed a test RMSE and does not judge whether the number is good. The general agent and the repair path were not used in this run.
- Keep your API key out of the repository. `.env` is listed in `.gitignore`, and so are the copies of `SNLS80mV.mat` inside `generated_scripts/`; the one in the project folder is the source.
