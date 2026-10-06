# Progress Report 1

**Project:** A Comparative Evaluation of Memory Architectures for LLM-Based Conversational Agents  
**Reporting date:** September 30, 2026

## 1. Project Overview

This project evaluates how three memory designs for conversational LLM agents—raw conversation history, vector retrieval, and a running summary—compare in answer accuracy, token use, latency, and failure modes. I am using LongMemEval, a benchmark of long, multi-session conversations and memory questions. The comparison is designed to hold the answering model, generation settings, answer instructions, and question set constant so that observed differences can be attributed as much as possible to the memory approach. The experiment design and comparison rules are recorded in the [research notes](research_notes.md#L1) and [configuration](../experiment_config.json#L1).

## 2. Work Completed

I created the project structure and local Python environment, documented the setup, and selected Ollama with `llama3.1:8b` as the local answering model. A real API request confirmed that the model responds and returns prompt and generation token counts; the captured request and response are in [model_check.json](../results/model_check.json#L1). The model was tested on an Apple M4 with 16 GB unified memory.

I downloaded and verified the cleaned LongMemEval dataset. Its recorded size is 277,383,467 bytes, and its SHA-256 checksum is recorded in the [experiment configuration](../experiment_config.json#L56). I wrote and ran [inspect_data.py](../src/inspect_data.py#L1), which checks question-ID uniqueness and alignment of each history with its dates and session IDs. The inspection found 500 questions, with 38–62 sessions per question (mean 47.734), and produced the [dataset summary](../results/dataset_summary.json#L1). I also generated a fixed 14-question pilot sample using seed 42; the selected IDs are saved in [pilot_ids.json](../results/pilot_ids.json#L1).

I implemented the raw-history pipeline in [baseline_raw_history.py](../src/baseline_raw_history.py#L1). It formats conversation turns with their dates, session IDs, and speaker roles, then removes the oldest complete turns until the prompt fits the context budget. I selected a 32,768-token context budget after measuring prompt-processing speed and GPU/CPU memory use on this machine. This is smaller than LongMemEval's approximately 115K-token setting and is a documented hardware/runtime limitation, not a directly equivalent reproduction of that setting.

The pipeline completed an end-to-end smoke test on one abstention question, `e5ba910e_abs`. The source history contained 522 turns; the pipeline retained 122 turns, measured 28,105 prompt tokens during fitting against a 32,256-token input allowance, and completed the overall test in about 285 seconds. The answer was a non-answer phrased as a general lack of access to personal shopping history. This confirms the pipeline can execute, but it is only one question and has not been graded systematically. The details are in the [development log](development_log.md#L49).

The full 14-question pilot has not yet been run or saved: `results/baseline_raw_history_predictions.json` is not present. The vector-retrieval and running-summary pipelines, common evaluation/measurement code, and comparative results are also not yet complete. Therefore, this report does not make claims about which memory design performs best.

## 3. Evidence of Progress

The project is currently local and has not been pushed to GitHub; there is no GitHub URL or commit to provide yet. The following repository artifacts document the work completed:

| Evidence | What it shows |
|---|---|
| [Experiment configuration](../experiment_config.json#L1) and [research notes](research_notes.md#L1) | Research question, planned memory approaches, controlled-comparison rules, model settings, and dataset provenance |
| [Data inspection script](../src/inspect_data.py#L1), [dataset summary](../results/dataset_summary.json#L1), and [pilot IDs](../results/pilot_ids.json#L1) | Dataset validation, measured session/question counts, and reproducible pilot selection |
| [Model check script](../src/check_model.py#L1) and [captured model check](../results/model_check.json#L1) | A successful local model request and runtime-provided token counts |
| [Raw-history pipeline](../src/baseline_raw_history.py#L1) and [development log](development_log.md#L49) | Implementation details, the one-question smoke test, measured truncation, and the debugging decisions behind the chosen runtime settings |

There are no screenshots or charts yet, and there are no saved pilot predictions to cite. The dataset itself is excluded from version control; its source revision and checksum are documented in the configuration file.

## 4. Challenges

**Installing and downloading the model.** Homebrew initially failed during its automatic update because it could not resolve a package-hosting domain. Retrying without the auto-update step succeeded. The 4.9 GB model download then repeatedly restarted; setting Ollama to use one transfer stream allowed it to complete.

**Choosing a practical context budget.** The larger context setting could load, but it moved part of the model onto the CPU. Also, the initially recommended flash-attention and quantized-KV-cache settings processed prompts at about 37 tokens per second, while Ollama's default settings measured about 107 tokens per second on the tested prompt. I chose a 32,768-token budget under the faster default settings to keep the model GPU-resident and the experiment practical. This choice is recorded in the configuration and should be treated as a limitation when interpreting results.

**Keeping token fitting tractable.** Counting tokens by repeatedly evaluating near-context-sized prompts made the first truncation search too slow. I replaced that approach with an inexpensive character-based estimate calibrated against the model's actual prompt-token count. A separate hang occurred when Ollama unloaded the model between calls; setting a 30-minute `keep_alive` on each request resolved it. The implementation still uses the model's returned token counts for recorded measurements rather than reporting the character estimate as an exact count.
