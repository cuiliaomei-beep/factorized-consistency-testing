# Tcc: Factorized Consistency Testing for AI Compilation

This repository is the artifact of the paper *Factorized Consistency Testing for AI Compilation*. It contains the tool, the experiment inputs, the experiment scripts, and the intermediate and final results.

An AI compiler such as `torch.compile` should preserve what eager execution does. A violation does not have to change the returned value. It can drop a state mutation, return an alias instead of a copy, change a gradient, skip a user-written exception handler, or crash when an artifact is destroyed.

Tcc tests this property by factorizing the test space into two groups of factors:

- **Program properties.** Shapes, dtypes, scalar values and flags that the compiled code reads. They are derived from the source and from operator metadata.
- **Compilation decisions.** Graph-break position, backend layer, execution context and artifact route.

Each test changes one factor against a reference execution. The two executions are compared on seven observables: value, metadata, identity and aliasing, exception type and routing, side effects, gradients, and process survival. Staged reruns then report the compiler layer at which a difference first appears.

## Repository layout

| Path | Content |
|---|---|
| `run.py` | Command-line entry point (self-checks, per-phase inspection, full pipeline, campaigns, baselines) |
| `tcc/` | The tool (see the module table below) |
| `tests/` | Offline tests that do not need torch |
| `scripts/` | Experiment scripts: differential sweeps (`*_diff.py`, `*_sweep.py`), candidate confirmation (`confirm_*.py`), root-cause diagnosis (`diag_*.py`), supplementary experiments (`gb_p01_*.py`, `oracle_tiers_*.py`, `p11_*.py`), issue status (`issue_status_report.py`) |
| `kaggle/` | Drivers and job definitions for the Linux and CUDA runs on Kaggle |
| `benchmark/` | Historical-issue benchmark: `historical_bugs.json` (67 verified issues: 35 fixed, 26 still present, 6 only in the old version) and their fix status on PyTorch 2.14 |
| `reproducers_gh/` | Experiment input: 193 stand-alone reproducers mined from the PyTorch issue tracker (the historical benchmark; see REPRODUCE.md for where it is used) |
| `reproducers_gh_cuda/`, `reproducers_gh_rerun/`, `reproducers_models/`, `reproducers_example/` | Experiment inputs: CUDA reproducers (336), reruns (13), programs built from torchvision architectures (68), and two examples. Each file defines `f` and `args` |
| `data/` | `findings.csv`, the classification of all findings (layer, symptom, factor, observable, status); `stats.py` computes the counts used in the paper |
| `results/` | Raw outputs, logs and summaries of every experiment run on Windows; see `results/README.md` |
| `results/SUPPLEMENTARY_EXPERIMENTS.md` | Summary of the supplementary experiments (recomputation, evidence audit, baselines, oracle tiers, cost) |
| `reports*/`, `gpu_reports*/` | Candidate reports per sweep: one directory per candidate with `issue.md`, `minimal.py`, expected and actual output, and execution trace |
| `kaggle_out/` | Outputs of the Kaggle jobs (Linux CPU and Tesla T4), one directory per job |
| `results0909/` | Results of the first full run (2026-09-09) |
| `REPRODUCE.md` | Step-by-step reproduction of every research question |

Modules of `tcc/`:

| Phase | Modules |
|---|---|
| 1. Compilation sites and factor derivation | `sites.py`, `ir.py`, `binding.py`, `binding_ast.py`, `factors.py`, `scs.py` |
| 2. Seeds, variation and execution | `seeds.py`, `corpus.py`, `program.py`, `generate.py`, `execute.py`, `runner.py` |
| 3. Observation, oracle, localization, minimization | `observe.py`, `oracle.py`, `localize.py`, `minimize.py` |
| 4. Reports and campaign ledger | `report.py`, `campaign.py` |
| Path-side and program-side analyzers | `aoti_diff.py`, `cache_diff.py`, `decomp_diff.py`, `dtype_matrix.py`, `metamorphic.py`, `sweep_common.py` |
| Python-semantics corpus | `dynamo_semantics*.py` (274 programs in five batches; the first three batches, 185 programs, are the frozen corpus of RQ1) |
| Evaluation | `faults.py` (injected faults), `benchmark.py`, `baselines.py`, `ablation.py`, `experiments.py`, `metrics.py`, `mine.py` |
| Support | `selfcheck.py`, `compat.py` |

## Requirements

- Python 3.14 (3.11 and 3.12 were used on Linux).
- PyTorch 2.14.0. Some experiments also use a nightly build of the 2.15 development branch.
- A C++ compiler for TorchInductor: MSVC on Windows, GCC on Linux. On Windows, `tcc.compat.ensure_msvc_env()` loads the MSVC environment automatically.
- CUDA experiments ran on a Tesla T4 on Kaggle (CUDA 13, Triton).

Exact package versions and install commands are in `REPRODUCE.md`.

## Quick start

From the repository root:

```bash
python run.py selfcheck --offline     # parts that do not need torch
python run.py selfcheck               # oracles, compile counters, layered localization (about one minute)
python tests/test_offline.py          # offline tests of the report and deduplication chain
```

The self-checks inject a backend that deliberately returns wrong results, so that every oracle must raise an alarm. They also run the real `aot_eager` backend, which must raise none. Treat no output as evidence until both pass.

Run the full pipeline on the built-in programs:

```bash
python run.py run --backend inductor --reruns 2 --minimize
```

## Reproducing the paper

`REPRODUCE.md` maps each research question, table and figure to the commands that produce it and to the files that hold its numbers:

- **RQ1:** additional detection from compilation-factor variation.
- **RQ2:** detection gains from expanded observations.
- **RQ3:** construction effectiveness and probe cost.
- **RQ4:** real findings and evidence strength.
- **Supplementary experiments:** see `results/SUPPLEMENTARY_EXPERIMENTS.md`.

Some numbers cannot be reproduced on a CPU-only Windows machine. These are the CUDA-only items, a Windows-only crash, and the status of reported GitHub issues, which keeps changing. `REPRODUCE.md` says which ones and how to rerun the Kaggle jobs.

## Notes

- Results were produced with PyTorch 2.14.0 between September 2026 and October 2026. Later PyTorch versions may already fix some of the reported defects, so a rerun can show fewer divergences.
- Files larger than 10 MB are stored gzip-compressed with the suffix `.gz`. Decompress them before use.
- Compiled binaries and object files of the C++ experiments are not included. The scripts regenerate them.
