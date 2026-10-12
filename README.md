# 448 Midterm Project: Who Wrote It?

Classifying Llama, Gemini, and Qwen responses with a convolutional neural network (CNN) and a recurrent neural network (LSTM).

**Team:** Adithya Vemulapali, Shreya Garg, Ani Ganesan.

## Results

Primary seed 42, 150 test responses (50 prompt groups), equal family representation:

| Model | Prompt only | Response only | Prompt + response |
|---|---:|---:|---:|
| CNN | 33.3% | 86.0% | 90.0% |
| LSTM | 33.3% | 60.7% | 68.0% |

The three identical copies of each prompt have different family labels, so a deterministic prompt-only classifier is correct on exactly one of three. Adding the prompt can still help interpret the response in context; it is not guaranteed to improve accuracy. The CNN's paired improvement interval excludes zero in this experiment; the LSTM's reaches zero.

We also ran two additional training seeds per architecture, a formatting-removal ablation, and held-out writing tasks: **14 neural runs total**. The domain test scores are 64.3% for CNN and 50.3% for LSTM. These are results for this particular dataset and model configurations, not reliable forensic attribution of arbitrary text.

## Start here

- [Results table](results/summary.csv): aggregate metrics for every experiment.
- [Tests](tests/): checks for data isolation, padding, and saved-run integrity.

## Setup and verification

Use Python 3.10 or later. The original training environment used Python 3.13 and PyTorch 2.14.1 on an M2 Pro GPU. The exact environment is also recorded in each run's metrics.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python verify_results.py --checkpoints
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. A CUDA-enabled PyTorch installation is needed for NVIDIA acceleration; CPU execution works too. `--device auto` selects CUDA, then Apple MPS, then CPU.

## Retrain without replacing the published evidence

The default suite command checks and skips the already-completed runs. Use a new results directory to perform actual fresh training:

```bash
python run_experiments.py --results-dir results_fresh --device auto
python analyze_results.py --results-dir results_fresh --figures-dir figures_fresh
python verify_results.py --results-dir results_fresh
```

To train one model:

```bash
python train_authorship_attribution.py --model cnn --text-source output --out results_fresh/my_cnn
```

New weights are saved under `models/<run-name>/model.pt`, which is excluded from Git. The original published weights under `checkpoints/` are not overwritten. Hardware and package differences can change exact numerical results even with the same seed.

To regenerate tables and figures from the original saved runs:

```bash
python analyze_results.py
```

## Classify a new response

Put the response text in `response.txt`, then run:

```bash
python predict.py --checkpoint checkpoints/cnn_output_s42.pt --output-file response.txt
```

The classifier always chooses among these three families. It cannot identify human authors or an unknown model. Its probability scores are uncalibrated.

## Dataset

The master file contains **500 prompts across 20 categories, 25 per category**. Each family has a response to every prompt.

| Family | Recorded model | Dataset | Responses |
|---|---|---|---:|
| Llama | `llama3.1:8b` | `ollama_outputs_llama3.1_8b.json` | 500 |
| Gemini | `gemini-3.8-flash` | `gemini_outputs_gemini_3.8_flash.jsonl` | 500 |
| Qwen | `qwen2.5:7b` | `ollama_outputs_qwen2.5_7b.json` | 500 |

Every record stores its prompt, output, family/model, prompt ID/category, requested settings, and timestamp. Classifiers see only the specified text fields, never metadata. Raw dataset files, checkpoints, and saved experiment results are unchanged from the reviewed project.

## Optional collection and development

Training uses the saved data and needs no API keys. The collection scripts are included for context and future collection. Their model registry reflects the original collection; current provider availability is not guaranteed.

```bash
pip install -r requirements-collection.txt  # Only for collecting new API responses
pip install -r requirements-dev.txt         # Optional formatter
python -m black --check .
```

API keys come from environment variables. Do not commit secrets or start collection against the supplied data files just to run the project. Collection event logs and partial timing metadata are retained in `provenance/`.

AI assistance was used in collection tooling, implementation, auditing, and organization. Team members should understand and verify the code and conclusions they submit.
