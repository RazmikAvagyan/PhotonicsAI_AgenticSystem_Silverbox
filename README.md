# Agentic system: Silverbox (Homework 1) and focusing through a simulated bench (Homework 2, Task 2)

A LangGraph system of four roles (orchestrator, scientist, general agent, executor). In Homework 1, Task 6 the roles plan,
write, run and report a Neural-ODE fit of the Silverbox benchmark (branch `main`). This branch (`hw2-task2`) adds a
simulated optical bench and a focusing tool, and the agent uses them to focus a beam through a scattering medium.
Models: Gemini through `langchain-google-genai` (`gemini-3.8-flash`, temperature 0).

The human-in-the-loop work of Task 1 is on the branch `hw2-task1`.

## The simulated bench (`bench.py`)

Laser, DMD, scattering medium, camera.

- The DMD has 32 x 32 micromirrors and changes only the amplitude of the input field: a mirror is on (1) or off (0).
- The scattering medium is one complex transmission matrix `T` (1024 x 1024, random entries, fixed seed), so every
  experiment sees the same medium.
- The camera has 32 x 32 pixels and records `|T E_in|^2 * (1 + 0.05 * e)`, with `e` a standard normal number per pixel.
  A second shot of the same pattern is therefore not a copy of the first.

## The focusing tool (`tools.py`, `run_focusing_tool`)

The search follows the algorithm of the assignment:

1. Divide the DMD into macropixels (blocks of `macropixel x macropixel` mirrors) and start from a random pattern.
2. After each transmission, read the intensity in the focus region (a `focus_size x focus_size` square at the centre of
   the camera) and in the proximal background (a ring of width `background_width`, one pixel away from the focus,
   split into `n_regions` sectors).
3. Enhancement = mean intensity in the focus / mean of the mean intensities of the background regions.
4. Flip `flips` random macropixels, take a new shot, keep the flip only if the enhancement increased, repeat `iterations` times.

The reported enhancement is computed from a new camera shot of the final pattern. The tool saves one figure per run
(random-pattern image, image after the search, enhancement during the search) and returns the enhancement.

| Setting | Allowed values | Default |
|---|---|---|
| `macropixel` | 1, 2, 4, 8, 16 | 4 |
| `focus_size` | 1 to 5 | 1 |
| `background_width` | 1 to 8 | 4 |
| `n_regions` | 2 to 8 | 4 |
| `iterations` | 50 to 20000 | 1000 |
| `flips` | 1 to 8 | 1 |
| `seed` | 0 to 1000000 | 0 |

## How the agent works

- The only request typed during the run is `Focus the beam.` The agent chooses all settings.
- The orchestrator asks the scientist for a plan (physics, effect of each setting, four experiments). It then gives the
  settings of one experiment at a time to the executor.
- The executor is the only role that calls the tool. It writes the settings of one experiment as a JSON object, and the
  tool returns the enhancement.
- The orchestrator finishes only after at least 4 experiments with different settings have returned an enhancement, or after
  12 decisions. It then names the best settings.

## Result of the run in this repository

| Run | macropixel | focus_size | background_width | n_regions | iterations | flips | seed | Enhancement |
|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 1 | 3 | 4 | 300 | 1 | 42 | 2.56 |
| 2 | 8 | 1 | 3 | 4 | 1200 | 1 | 42 | 8.36 |
| 3 | 4 | 1 | 3 | 4 | 5000 | 1 | 42 | 27.57 |
| 4 | 8 | 1 | 3 | 4 | 1200 | 2 | 42 | 8.43 |

## Files

| File | Content |
|---|---|
| `bench.py` | simulated bench, regions, enhancement, focusing search and figure |
| `tools.py` | `run_focusing_tool` (limits on the settings) and the coding tool of Homework 1 |
| `agents.py` | prompts and model calls of the four roles (executor returns the settings as JSON) |
| `nodes.py` | graph nodes, parsing of the orchestrator reply, finish guard (at least 4 experiments) |
| `main.py` | task text, graph, run loop, table of experiments, `--resume` |
| `run_task2_log.txt` | dialogue of the run above |
| `experiments_table.md` | table of the experiments of the run above |
| `generated_scripts/run_20261012_204001/focus_1.png` to `focus_4.png` | camera images of the four runs |
| `videos/task2_focusing_run.mp4` | screen recording of the run |

The Silverbox scripts of Homework 1 (`data.py`, `train.py`, `evaluate.py`, `model.py`, `integrator.py`) are unchanged.

## Setup

1. Python 3.10 with `langgraph`, `langchain-google-genai`, `python-dotenv`, `numpy`, `matplotlib`.
2. A file `.env` with `API_KEY = <your Gemini key>` (not committed).

## Run

```
python main.py
```

The dialogue is printed turn by turn. At the end the program prints the table of experiments and writes
`dialogue_log.txt` and `experiments_table.md`. The images are saved in `generated_scripts/run_<date>_<time>/`.

If the run stops early (for example when the API quota ends), `python main.py --resume` continues the run saved in
`dialogue_log.txt`.