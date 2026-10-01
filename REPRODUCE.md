# REPRODUCE.md - step-by-step reproduction guide

This guide reproduces the evaluation of the paper *Factorized Consistency Testing for AI Compilation* (tool name: Tcc), FSE'27
submission, revised version of 2026-10-01. Section, table, figure and research-question numbers below are the
paper's. All commands are run from the artifact root (the directory that contains `run.py`), unless stated otherwise.

Contents

1. Artifact layout and naming
2. Environment setup
3. Self-checks
4. Paper-to-artifact map (quick reference)
5. Section 4.1: setup, Table 3, Table A1 (injected faults)
6. RQ1 (Section 4.2, Table 4)
7. RQ2 (Section 4.3, Table 5)
8. RQ3 (Section 4.4: Table 6, Figure 5, Table 7, layer diagnosis, Table 8)
9. RQ4 (Section 4.5: Table 9, case studies, C37, other targets, Table 10)
10. Section 4.6 Discussion (supplementary exploration, Table 11)
11. Supplementary experiments of 2026-09-25 (P0-1, P0-2, P1-1, P1-2, P1-3)
12. Result directory -> experiment
13. What cannot be reproduced locally; re-running the Kaggle jobs
14. Known gaps in this artifact

---

## 1. Artifact layout and naming

| Path | Content |
|---|---|
| `tcc/` | the tool (static analysis, test construction, execution, oracles, localization, reports, injected faults, experiment drivers) |
| `run.py` | command-line entry point (`python run.py --help`) |
| `scripts/` | stand-alone experiment, sweep, triage and confirmation scripts |
| `tests/test_offline.py` | offline unit checks |
| `kaggle/` | Kaggle job driver (`drive.py`), packer (`pack.py`), job scripts (`jobs/`), job generators (`make_*.py`) |
| `benchmark/` | historical-bug manifest (`historical_bugs.json`) and fix-status data (`fix_status_2.14.json`) |
| `reproducers_*/` | input programs (issue reproducers mined from the PyTorch tracker, torchvision models, examples) |
| `results/`, `results0909/`, `reports*/`, `gpu_reports*/`, `kaggle_out/` | recorded outputs (Section 12) |
| `data/findings.csv`, `data/stats.py`, `data/stats.json` | classification of the 73 filed report items and the counting script |

Internal names differ from the paper's research-question numbers. The tool was written against an earlier
plan, so file names such as `rq1_*`, `rq2_*` do not match the paper's RQ numbers:

| File / function in the artifact | Paper item |
|---|---|
| `run.py bench`, `results/rq1_benchmark.json` (`tcc.benchmark`) | full Tcc run on the 19 injected faults; layer diagnosis (Section 4.4.2); Table A1 assignment counts |
| `run.py baselines`, `results/rq2_baselines.json`, `results/RQ2.md` | Table 6 (RQ3 primary comparison) and the NNSmith rows of Table 8 |
| `run.py ablate`, `results/rq3_ablation.json` | Figure 5 (RQ3 cumulative ablation) |
| `run.py cache`, `results/rq4_cache.json`, `results/RQ4.md` | Table 7 (RQ3 history validation) and the guard-switch row of Table 11 |
| `scripts/gb_p01_audit.py`, `results/p01_audit/`, `results/p13_timing/` | Table 4 (RQ1) |
| `scripts/oracle_tiers_real.py`, `results/p12_oracle_tiers/` | Table 5 (RQ2) |
| `data/findings.csv`, `results/p02_audit/`, `kaggle_out/chenyuefei_notebook*/` | RQ4 (Table 9, Table 10) |

**Important: most commands write into `results/` by default and some resume from existing output.**
To reproduce without overwriting the shipped numbers, either pass a separate output directory where the command
has one (`--out results_repro/...`), or copy the shipped directory away first. Resumable sweeps (those writing
JSONL "one line per case") skip work items already present in their output file; move the old file away to re-run
from scratch.

---

## 2. Environment setup

### 2.1 Versions used for the paper

| Component | Version |
|---|---|
| OS | Windows 11 Pro for Workstations 10.0.26200 (x86-64), no GPU |
| Python | 3.14.7 |
| PyTorch | 2.14.0+cpu (release); development builds 2.15.0.dev (nightly) for reproducer re-checks |
| C++ compiler for TorchInductor | MSVC 14.44.35207 (Visual Studio 2022 Build Tools, "Desktop development with C++") |
| numpy / mpmath / sympy | 2.5.2 / 1.3.0 / 1.14.0 |
| nnsmith | 0.1.0 (NNSmith baseline and TorchProbe) |
| expecttest | 0.3.0 (needed by the OpInfo database used by OpInfo-based sweeps) |
| torchvision | 0.29.0+cpu (only for the torchvision-model campaign; installed with `--no-deps`) |
| Linux CPU and CUDA | Kaggle notebooks: Linux x86-64, Python 3.12, PyTorch 2.10.0 and 2.14.0 (CPU), Tesla T4 with 2.10.0+cu128 / 2.14.0+cu130 |

### 2.2 Install (main environment)

```
python -m venv venv
venv\Scripts\activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy mpmath nnsmith==0.1.0 expecttest hypothesis certifi psutil
pip install --no-deps torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu   # optional
```

Third-party imports used by the code, and where:

| Package | Needed by |
|---|---|
| `torch` | everything except `run.py selfcheck --offline` and `tests/test_offline.py` |
| `numpy` | OpInfo sweeps, `scripts/oracle_tiers_*.py`, `scripts/xtarget_numpy_diff.py`, `scripts/ort_opt_diff.py` |
| `mpmath` | `scripts/special_vs_mpmath.py`, `scripts/binary_vs_mpmath.py`, `scripts/oracle_tiers_cases.py` (multi-precision references) |
| `nnsmith` | `run.py baselines` (NNSmith legs), `run.py run/campaign --nnsmith N`, `scripts/p11_nnsmith_seeds.py`, TorchProbe |
| `expecttest`, `hypothesis` | the OpInfo database (`torch.testing._internal`) used by OpInfo-based commands |
| `certifi` | read-only GitHub queries (`tcc/mine.py`, `scripts/fix_in_release.py`, `scripts/pr_driven_diff.py`) |
| `psutil` | `tcc/metrics.py` (`run.py metrics`) |
| `torchvision` | `scripts/make_model_reproducers.py`, `scripts/confirm_model_bf16_grad.py` |
| `numba`, `jax`, `onnx`, `onnxruntime` | only the cross-target probes (Section 9.4); use a separate environment (2.4) |
| `kaggle` | only `kaggle/drive.py` and `kaggle/kernel_log.py` (Section 13) |

### 2.3 Windows: MSVC for TorchInductor

TorchInductor's CPU backend compiles generated C++ with the host compiler. On Windows this must be `cl.exe`.
`run.py` and most scripts call `tcc.compat.ensure_msvc_env()` at start-up: it finds the newest Visual Studio
installation with `vswhere.exe`, runs `vcvars64.bat` and imports `PATH`, `INCLUDE`, `LIB` and related variables
into the process. Nothing needs to be done by hand if Visual Studio 2022 (or its Build Tools) with the C++
workload is installed. If `cl` is not found, Inductor fails with `InvalidCxxCompiler: Compiler: cl is not found`;
in that case open an "x64 Native Tools Command Prompt" and run the commands from there.

### 2.4 Optional environments

* **Nightly build** (reproducer re-checks on the development branch). A separate virtual environment next to the
  artifact root was used, with a CPU nightly (`2.15.0.dev20260921+cpu` for most re-checks):

  ```
  python -m venv ..\venv_nightly
  ..\venv_nightly\Scripts\python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
  ..\venv_nightly\Scripts\python -m pip install numpy mpmath expecttest hypothesis onnx onnxruntime onnxscript
  ```

  Scripts are then run with that interpreter, for example
  `..\venv_nightly\Scripts\python scripts/verify_0913_issues.py`. Nightly wheels of a given date are removed from
  the index after some time, so the exact nightly builds cannot be re-installed; use the current nightly.
* **Cross-target environment** (Numba / JAX / ONNX Runtime probes): numpy 2.5.3, numba 0.67.0, jax and jaxlib
  0.11.2, onnx 1.23.0, onnxruntime 1.30.0:

  ```
  python -m venv ..\venv_xtarget
  ..\venv_xtarget\Scripts\python -m pip install numpy numba jax jaxlib onnx onnxruntime
  ```

* **Linux CPU and CUDA** runs were made on Kaggle (Section 13).

---

## 3. Self-checks

Run these first. Nothing produced by the pipeline counts as evidence until they pass.

```
python run.py selfcheck --offline
python run.py selfcheck
python tests/test_offline.py
```

| Command | What it checks | Expected outcome | Time |
|---|---|---|---|
| `python run.py selfcheck --offline` | static analysis derives the boundaries, facts and dedup signatures promised by the design (no torch execution) | `10/10 checks passed` | about 4 s |
| `python run.py selfcheck` | the above plus: exact compile counting; each oracle fires on the injected fault built for it and stays silent on the real compiler (`aot_eager`); fp64 reference; staged localization; an Inductor cold run actually executes | `29/29 checks passed` (recorded in `results/selfcheck_torch.log`) | about 15 s |
| `python tests/test_offline.py` | offline unit checks (IR, factors, report rendering, oracle helpers); writes only to a temporary directory | `38/38 offline checks passed` | about 2 s |

The last torch check (`inductor cold run executes`) fails if MSVC is not available (Section 2.3).

---

## 4. Paper-to-artifact map (quick reference)

| Paper item | Command (Section of this guide) | Numbers are in |
|---|---|---|
| Table 4 (RQ1) | `scripts/gb_p01_audit.py`, `--timing 3` (6) | `results/p01_audit/audit_eager.json`, `results/p13_timing/audit_eager.json` |
| Table 5 (RQ2) | `scripts/oracle_tiers_real.py run / control-corpus / control-autocast / report` (7) | `results/p12_oracle_tiers/tiers.csv`, `summary.json` |
| Table 6 (RQ3) | `run.py baselines`; `scripts/p11_random_seeds.py` (8.1) | `results/rq2_baselines.json`; `results/p11_baselines/random_seeds_summary.json` |
| Figure 5 | `run.py ablate` (8.2) | `results/rq3_ablation.json` |
| Table 7 | `run.py cache` (8.3) | `results/rq4_cache.json` |
| Layer diagnosis | `run.py bench` (8.4) | `results/rq1_benchmark.json` |
| Table 8 | `run.py baselines`, `scripts/p11_nnsmith_seeds.py`, `scripts/p11_torchprobe_*.py` (8.5) | `results/p11_baselines/` |
| Table 9 | `data/stats.py` + tracker status (9.1) | `data/findings.csv` (partly; see 9.1) |
| C37 (1,345 tests) | `scripts/opinfo_edge_sweep.py --unsupported` on Kaggle (9.3) | `kaggle_out/edgeunsup/cases.jsonl` |
| Numba / JAX / MSVC / ONNX Runtime | `scripts/xtarget_numpy_diff.py`, `scripts/cxx_intrinsic_algebra_diff.py`, `scripts/ort_opt_diff.py` (9.4) | `results/xtarget/`, `results/cxx_algebra/`, `results/ort/` |
| Table 10 | Kaggle jobs `bench210`, `bench214`, `scripts/gh_summary.py` (9.5) | `kaggle_out/chenyuefei_notebook*/plan/reports_gh_cpu_2.*/KNOWN_BUGS.md`, `results/p02_audit/historical_pairs.csv` |
| Table 11 | `run.py diskcache`, `run.py cache`, `scripts/export_roundtrip_edge.py` (10.2) | `results/diskcache*/`, `results/rq4_cache.json`, `results/export_roundtrip/float32.jsonl` |
| Table A1 | `tcc/faults.py`; `run.py bench` (5) | `results/rq1_benchmark.json` (`entries`) |

Figures 1-4, Listing 1 and Tables 1-2 are conceptual or illustrative; they contain no measured data.

---

## 5. Section 4.1: setup, Table 3, Table A1

* **Implementation size** ("approximately 25,000 lines"): the Python files under `tcc/`, `scripts/`, `run.py`.
* **Table 3** summarizes the four RQs; its numbers are the ones of Tables 4-10 below.
* **Table A1 (19 injected fault models, 51 assignments).** The faults are defined in `tcc/faults.py`, in the order
  F01-F19 of Table A1: `codegen_value` (F01), `codegen_boundary_32` (F02), `codegen_boundary_31` (F03),
  `codegen_mod16_tail` (F04), `codegen_index_last` (F05, leading size 1), `codegen_noncontig` (F06),
  `codegen_fp16_rounding` (F07), `codegen_dynamic_flag` (F08), `codegen_scalar_edge` (F09),
  `functionalize_drop_mutation` (F10), `functionalize_alias_to_copy` (F11), `capture_swallow_exception` (F12),
  `autograd_wrong_gradient` (F13), `underspec_dtype` (F14), `underspec_scalar` (F15), `underspec_layout` (F16),
  `underspec_requires_grad` (F17), `cache_stale_on_return` (F18), `cache_result_memo` (F19). Each `Fault(...)`
  lists its programs (`programs=[...]`) from the 29-program corpus in `tcc/corpus.py` (`python run.py list`).
  The #P column is the number of `entries` per `fault` in `results/rq1_benchmark.json` (51 entries in total).
  The six history faults are F14-F19.

---

## 6. RQ1: additional detection from compilation-factor variation (Section 4.2, Table 4)

**What it measures.** The 185 programs of the frozen Python-semantics corpus (first three batches:
`tcc/dynamo_semantics.py` 77, `tcc/dynamo_semantics_more.py` 74, `tcc/dynamo_semantics_batch3.py` 34) run in one
process with one oracle in three arms: eager, compiled original program, compiled program with
`torch._dynamo.graph_break()` inserted after every top-level statement (AST rewrite). Two calls per program give
370 paired calls per arm. Backend `eager` (TorchDynamo only).

**Commands.**

```
python scripts/gb_p01_audit.py --backend eager --out results_repro/p01_audit
python scripts/gb_p01_summary.py --backend eager --out results_repro/p01_audit
python scripts/gb_p01_audit.py --backend aot_eager --out results_repro/p01_audit
python scripts/gb_p01_audit.py --backend eager --timing 3 --out results_repro/p13_timing
python scripts/gb_p01_summary.py --backend eager --out results_repro/p13_timing
```

Runtime recorded: 48.7 s (`eager`), 68.2 s (`aot_eager`), 188.5 s with `--timing 3`.
`gb_p01_summary.py` renders `P0-1_<backend>.md` next to the JSON; with a timing run it adds the paired timing section.
`gb_p01_audit.py` reloads the third batch so that the frozen 185-program version is used even though the fourth
batch redefines one program name (`chained_comparison_tensor_scalar`).

**Shipped outputs.** `results/p01_audit/audit_eager.json`, `audit_aot_eager.json`, `pairs_eager.csv`,
`pairs_aot_eager.csv`, `audit_*.log`, `minimal/`; timing: `results/p13_timing/audit_eager.json`,
`pairs_eager.csv`, `timing_eager_run1.log`.

**Reading Table 4** (key paths are inside `summary` of the JSON):

| Table 4 row | Original | Break policy | Source |
|---|---|---|---|
| Divergent calls (of 370) | 19 | 25 | `p01_audit/audit_eager.json`: `baseline_divergent_calls`, `break_divergent_calls` |
| Calls divergent only in this arm | 2 | 8 | `pairs.gone`, `pairs.new` (members in `pair_members`) |
| Programs with newly divergent calls | - | 4 | distinct program names in `pair_members.new` (`gen_send`, `gen_throw`, `gen_return_value_stopiteration`, `collections_types`) |
| Additional candidate roots | - | 2 | hand assignment in `scripts/gb_p01_summary.py`: C41 = `collections_types`; C42 = the three generator programs |
| Compiled frames | 614 | 1,175 | `frames_total_baseline`, `frames_total_break` |
| Median wall time (s), range | 13.23 (13.09-13.59) | 32.65 (32.20-32.97) | `p13_timing/audit_eager.json`: `timing_baseline` / `timing_break` -> `total_median_s`, `total_min_s`, `total_max_s`, `total_per_run_s` |

Other numbers of Section 4.2: 17 overlapping divergences = `intersection_calls`; 13 retained and 4 changed =
`pairs["same-divergence"]`, `pairs.changed`; 343 agreeing calls = `pairs.same`; frame ratio 1,175/614 = 1.91 and
time ratio 32.65/13.23 = 2.47. "Both recurring with aot_eager": `audit_aot_eager.json` has the same eight `new`
members. The unstable `deepcopy_module_state` program explains why the timing run shows 18/24 divergent calls
instead of 19/25 (`p13_timing/audit_eager.json`). Static insertion positions (477) and breaks actually hit (544) are
`static_insertion_positions` and `user_breaks_actually_hit_total`.

Minimized reproducers: `results/p01_audit/minimal/draft40_block0.py` (C41, no graph break needed) and
`draft41_block0.py` (C42), each with its recorded output `*.out.txt`.

The original discovery run (same corpus, script that rewrites programs and compares with an in-process baseline) is

```
python scripts/graph_break_insertion_diff.py --backend eager
```

It writes `results/graph_break_insertion/<backend>/report.json` (fixed location; the shipped `eager/report.json` is a
later run over the 274-program corpus, see Section 10.1). Logs of earlier runs: `results/graph_break_insertion_*.log`.

---

## 7. RQ2: detection gains from expanded observations (Section 4.3, Table 5)

**What it measures.** For each independent real root cause of the pipeline findings, the stand-alone reproducer
is executed once per side (eager, compiled) in a child process. Every observable is stored in one trace, and three
cumulative oracle tiers are evaluated offline on the same trace: T1 values and ordinary crashes or exceptions, T2
adds metadata, T3 adds aliasing/identity, state, gradients, exception routing, multi-precision reference and
process survival. Root causes and sub-cases are registered in `scripts/oracle_tiers_cases.py`.

**Commands.** The script has no `--out` option; it always writes `results/p12_oracle_tiers/`. Copy that directory
away before re-running.

```
python scripts/oracle_tiers_real.py run --workers 3
python scripts/oracle_tiers_real.py control-corpus
python scripts/oracle_tiers_real.py control-autocast --acdtype bfloat16 --shard 0/2
python scripts/oracle_tiers_real.py control-autocast --acdtype bfloat16 --shard 1/2
python scripts/oracle_tiers_real.py control-autocast --acdtype float16 --shard 0/2
python scripts/oracle_tiers_real.py control-autocast --acdtype float16 --shard 1/2
python scripts/oracle_tiers_real.py report
```

* `run` executes 125 sub-cases with 3 workers (B5 is skipped: it needs CUDA); recorded wall time about 1,580 s
  (`results/p12_oracle_tiers/run_all.log`). `--only C8,C41` restricts to root causes, `--primary-only` to the
  minimal triggers.
* `control-corpus` runs the 29 unmodified corpus programs (expected: consistent).
* `control-autocast` re-runs the 462 operators of the autocast sweep (`results/autocast/<acdtype>.jsonl`) once each
  under bfloat16 and float16 autocast: 462 x 2 = 924 samples, about 210 s per shard.
* `report` re-evaluates all saved traces offline and rewrites `REPORT.md`, `tiers.csv`, `root_causes.csv`,
  `subcases.csv` and `summary.json`. It is cheap and can be used to check the shipped traces without re-running.
* `eval <trace>` evaluates one trace file.

**Reading Table 5.** `tiers.csv` has one row per root cause (52 rows) with columns `T1`, `T2`, `T3` (0/1),
`weakest_tier` and `reproducible_here`. The paper's denominator of 50 roots excludes B5 (`reproducible_here` =
`no`, needs CUDA) and B13 (expected behavior; trace `traces/B13__00_aoti_lifted_constant_mutation.json`).
Over the remaining 50 rows: T1 = 35, T2 = 37, T3 = 50; added roots per tier = 2 (`weakest_tier` = T2: B15, B25)
and 13 (`weakest_tier` = T3). The 126 traces are the 127 files in `traces/` minus the B13 trace.
`summary.json` gives the same per tier including B13 (`detected_all`: T1 36, T2 38, T3 51;
`weakest_hist_all`).

Controls: `summary.json` -> `corpus_control` (29 programs, `detected` false for every tier: 0/29) and
`autocast_control_n` = 924 with `autocast_alarms` (two bfloat16 alarms, `nn.functional.bilinear` and
`linalg.polar`, raised by T1 and therefore by all tiers: 2/924). Per-sample control data:
`control_traces/` and `control_autocast_*.jsonl`.

Note: the primary sub-case of B7 reads the first code block of an issue draft (`0913issues/07-...md`, resolved
relative to the parent of the artifact root), which is not included. That sub-case fails in the artifact; the other
B7 sub-cases run.

---

## 8. RQ3: construction effectiveness and probe cost (Section 4.4)

The injected-fault experiments were run with `scripts/run_experiments.ps1` (and `scripts/run_rest.ps1`), which
call the `run.py` subcommands below with `--backend eager` and write the logs `results/rq*.log`. The commands
below add `--out` so that the shipped JSON files are not overwritten.

### 8.1 Table 6: factorized versus random construction

```
python run.py baselines --backend eager --budget 12 --out results_repro
```

Recorded wall time: 875 s for all legs. Output: `rq2_baselines.json`, `RQ2.md`, console log as in
`results/rq2_baselines.log`.

| Table 6 row | Source (`results/rq2_baselines.json`) |
|---|---|
| Tcc / full: 19 detected, 235 probes, 16 rejected | `B5_ours`: `detected`, `tests_executed`, `tests_invalid` |
| Random / full: 14, 355, 109 | `B1_random_full_oracle`: same keys (one seed, `runs` = 1) |
| Random / basic, 10 seeds: 10.4, 492.7, 107.5 | `results/p11_baselines/random_seeds_summary.json` -> `stats`: `detected.mean`, `tests_executed.mean`, `tests_invalid.mean`; per seed in `random_seeds.csv` |

Derived numbers: 120 fewer probes = 355 - 235 (33.8%); 93 fewer rejected contexts = 109 - 16 (85.3%).
Median 11, range 6-12 and sample standard deviation 1.65 are in `stats.detected`. The `B1_random` entry of
`rq2_baselines.json` is the mean over seeds 0-2 only (11.33), the value used in earlier drafts. Faults never found by
the basic-oracle random runs are in `random_seeds_per_fault.csv`.

The ten basic-oracle random seeds were produced by:

```
python scripts/p11_random_seeds.py --seeds 5 --budget 12 --out results_repro/p11
python scripts/p11_random_seeds.py --first-seed 5 --seeds 5 --budget 12 --out results_repro/p11/seeds5to9
python scripts/p11_merge_random_seeds.py results_repro/p11 results_repro/p11/seeds5to9 --out results_repro/p11
python scripts/p11_random_seeds_md.py --dir results_repro/p11
```

Recorded: 253 s (seeds 0-4) and 288 s (seeds 5-9). `--check-fixed` additionally runs the unmodified compiler per seed.

### 8.2 Figure 5: cumulative ablation A0-A6

```
python run.py ablate --backend eager --budget 12 --out results_repro
```

Recorded wall time about 170 s (sum of `wall_s` over the levels). In `results/rq3_ablation.json` -> `levels` ->
`A0` ... `A6` (and `FULL`, identical to A6 plus the fixed-compiler pass):

* panel (a) detected faults = `detected` (7, 8, 8, 10, 13, 13, 19);
* panel (b) executed probes = `tests_executed` (338, 350, 358, 343, 262, 244, 235) and rejected contexts =
  `tests_invalid` (109, 71, 71, 69, 69, 16, 16).

The level definitions are in the docstring of `tcc/ablation.py`. `false_positives_on_fixed` is 0 for every level.

### 8.3 Table 7: history-dependent faults

```
python run.py cache --backend eager --budget 12 --out results_repro
```

In `results/rq4_cache.json` -> `settings`: `ordinary_eager_vs_compiled` (cold calls), `random_sequence`,
`scs_guided_sequence`; columns: `detected` (of `faults` = 6), `tests` (executed probes: 148, 155, 93),
`mean_tests_to_trigger` (6.45 and 4.76, printed as 6.5 and 4.8). The six faults are listed in `faults`.
Wall time was not recorded separately.

### 8.4 Layer diagnosis (Section 4.4.2) and the full Tcc run

```
python run.py bench --backend eager --budget 16 --out results_repro
```

Recorded: 48.8 s for the fault loop plus a pass over the 19 programs with the unmodified compiler. In
`results/rq1_benchmark.json`: `detected` = 19 of `faults` = 19; `entries` (51 fault-program assignments) carry
`stage`, the first discrepant layer reported by the localization (`inductor_codegen`, `specialization_cache`,
`cache_invalidation`), to be compared with the injection location of each fault in `tcc/faults.py`;
`false_positives_on_fixed` = 0 with the results of the fixed pass under `fixed`. Console log: `results/rq1_bench.log`.

### 8.5 Table 8: generated-program exposure (NNSmith, TorchProbe)

| Table 8 row | Programs/runs/detected/probes | Source |
|---|---|---|
| Tcc | 19 / 1 / 19 / 235 | `results/rq2_baselines.json` -> `B5_ours` |
| NNSmith | 8 / 3 / 5 / 152 | `results/p11_baselines/nnsmith_seeds_summary.json` (6/19 per seed as counted by the harness) and `nnsmith_native_seed*.json`; minus one wrapper artifact (`codegen_scalar_edge` raising on a tensor argument, see `TORCHPROBE.md` section 6) = 5 |
| TorchProbe | 35 / 1 / 6 / 665 | `results/p11_baselines/torchprobe_setting_b_subfunction.json` (`tests_executed` 665, harness count `detected` 15) minus the wrapper artifact and the Dynamo `dictionary changed size during iteration` race = 6; per-fault breakdown in `TORCHPROBE.md` section 6 |
| NNSmith hybrid with Tcc contexts: 9 | | `results/rq2_baselines.json` -> `B3_nnsmith_plus_our_contexts` |

Other numbers of Section 4.4.3: "expresses three of 29 programs and none of the 185" =
`results/p11_baselines/torchprobe_applicability_summary.json` (`corpus29.representable`,
`dynsem185.representable`); "222 tests without a discrepancy" = 175 tests in
`torchprobe_run_n10_seeds0to34.json` plus 47 in `torchprobe_run_n20_seeds100to109.json`.

Commands:

```
python scripts/p11_nnsmith_seeds.py --seeds 3 --models 8 --budget 12 --out results_repro/p11
python scripts/p11_torchprobe_applicability.py --out results_repro/p11
```

The NNSmith seed run took about 330 s per model seed. TorchProbe needs extra setup (Section 11.3).

---

## 9. RQ4: real findings and evidence strength (Section 4.5)

### 9.1 Table 9 and the finding counts

`data/findings.csv` has one row per filed report item (73 rows): `id` (B1-B27 first round, C1-C46 second round),
`kind` (`issue` 55, `comment` 15, `feedback` 3 = MSVC Developer Community reports), `target`, `ref` (issue or
feedback number), `layer`, `symptom`, `axis`, `oracle`, `attribution`, `status`. The column legend is in the
docstring of `data/stats.py`.

```
python data/stats.py
```

prints every count and rewrites `data/stats.json` (`kind`, `status`, `attribution`, `layer`, `symptom`, `axis`,
`oracle`, the A-attributed subsets, `fixed_ids`).

How Table 9 relates to these files:

* **NewReports column** (PyTorch 53, Numba 1, JAX 1, MSVC 3, total 58) = rows with `kind` in {`issue`,
  `feedback`} grouped by `target`.
* **Confirmed, Fixed, Pending, Rejected** use the tracker state of 2026-10-01. `findings.csv` stores an older
  snapshot (2026-09-23; over all 73 items: fixed 14, confirmed 10, pending 48, rejected 1). The 2026-10-01
  classification is not stored in any shipped file, so these four columns cannot be recomputed from the artifact.
* **"13 known PyTorch defects"** (71 findings = 13 known + 58 new): the list of the 13 known defects is not stored in
  a shipped file; it cannot be derived from `findings.csv` alone.

Tracker state changes over time. A read-only refresh (GitHub CLI `gh` must be logged in; it only reads) is

```
python scripts/issue_status_report.py
```

It writes `results/issue_status/raw.json` (state, labels, human comments, linked pull requests of the 55 issues) and
prints one line per issue; the shipped `raw.json` and `report_2026-09-25.log` are the 2026-09-25 refresh. The step
that sorts items into the four status classes was a manual review of this output and is not scripted in the artifact.

### 9.2 Case studies

* **C41 (dropped `OrderedDict.move_to_end`)**: RQ1 data (Section 6); minimized reproducer
  `results/p01_audit/minimal/draft40_block0.py`; oracle trace `results/p12_oracle_tiers/traces/C41__*.json`
  (weakest tier T3, `state`).
* **adaptive_max_pool3d with channels_last_3d, wrong gradient (#197888)**: reproducers in `reports_layout/`
  (`pool_backward_noncontiguous_indices.py`, `compiled_pool_grad_channels_last.py`,
  `adaptive_max_pool3d_chlast_grad.py`), found by the layout and alias sweep
  `python scripts/layout_alias_sweep.py --isolate` (results in `results/layout_alias/`; summary with `--report`).

### 9.3 Rejected report C37 (eager dtype coverage): 1,345 CPU operator-dtype tests

Run on Kaggle (Linux CPU, torch 2.14.0) by job `kaggle/jobs/edgeunsup.py`, which runs

```
python scripts/opinfo_edge_sweep.py --unsupported --dtypes int8,uint8,int16,int32,int64,bool,float16,bfloat16,float32,float64,complex64 --out cases.jsonl
```

Output: `kaggle_out/edgeunsup/cases.jsonl` (one JSON line per (op, dtype); log `tcc-edgeunsup.log`, about 1,770 s).
Summary:

```
python scripts/opinfo_edge_sweep.py --report kaggle_out/edgeunsup/cases.jsonl
```

It prints `1345 (op, dtype) pairs; verdicts: {'DIFF': 445, 'NO_SAMPLES': 55, 'ok': 845}`. Rows with `verdict` = `DIFF:SILENT` (eager raises, compiled returns) cover
134 operators. The file contains 443 such rows; the paper's 418 could not be re-derived from the shipped files
(see Section 14). Grouping into five mechanisms: `results/pairs/unsupported_family_list.md`; representative
members: `reports_pairs/unsupported_dtype_family.py`. The same sweep can run locally on Windows (drop `--out` to use
the default `results/opinfo_edge/cases_0.jsonl`).

### 9.4 Applicability evidence from other targets

* **Numba and JAX (268 NumPy-style programs; C7, C12).** One interpreter per target:

  ```
  python scripts/xtarget_numpy_diff.py --target dynamo --isolate
  ..\venv_xtarget\Scripts\python scripts/xtarget_numpy_diff.py --target numba --isolate
  ..\venv_xtarget\Scripts\python scripts/xtarget_numpy_diff.py --target jax --isolate
  python scripts/xtarget_numpy_diff.py --report
  ```

  Outputs `results/xtarget/<target>.jsonl` (268 lines each; also `numba_parallel`, `numba_fastmath`). Minimal
  reproducers: `reports_xtarget/numba_intmin_remainder_crash.py` (C7), `reports_xtarget/jax_gcd_intmin_hang.py` and
  `scripts/diag_jax_gcd_hang.py` (C12). Linux re-check: Kaggle job `xtargetlinux`.
* **MSVC (C6, C18, C19).** Optimization-level differential over SIMD intrinsic compositions:

  ```
  python scripts/cxx_intrinsic_algebra_diff.py --compiler msvc
  python scripts/cxx_intrinsic_algebra_diff.py --compiler msvc --report
  ```

  Outputs `results/cxx_algebra/msvc/`, `msvc_v1_diffs.jsonl`, `msvc_v2_diffs.jsonl`; `gcc` / `clang` variants via
  Kaggle job `cxxalgebra`. Grouping into families: `scripts/cxx_algebra_families.py`. Reproducers:
  `reports_rewrites/msvc_abs/` (C6, `abs(abs(x))` erased under `/O1`, `/O2`), `msvc_addzero/`, `msvc_blend/`. The
  Inductor-level host-compiler differential is `scripts/host_opt_diff.py --shard i/3` and `--report`
  (`results/host_opt/`).
* **ONNX Runtime (154 programs).** Needs onnx and onnxruntime (nightly or cross-target environment):

  ```
  ..\venv_nightly\Scripts\python scripts/ort_opt_diff.py --isolate
  python scripts/ort_opt_diff.py --report
  ```

  `--report` prints `154 programs; {'BOTH': 24, 'EXPORT_FAIL': 7, 'ok': 123}` from `results/ort/cases.jsonl`:
  seven export failures, 123 agree with eager, 24 show the same anomaly at both optimization levels.

### 9.5 Table 10: historical scripts on PyTorch 2.10 and 2.14

193 issue reproducers (`reproducers_gh/`, manifest `reproducers_gh/MANIFEST.json`) were run on Kaggle Linux CPU
with the same package and settings on torch 2.10.0+cpu and 2.14.0+cpu. Job scripts: `kaggle/jobs/bench210.py`,
`kaggle/jobs/bench214.py`. Each installs the torch version and runs

```
python run.py campaign --from-dir reproducers_gh --backend inductor --reruns 2 --max-contexts 12 --test-budget 16 --time-budget 180 --isolate --child-timeout 900 --out reports_gh_cpu_<version>
python scripts/gh_summary.py reproducers_gh reports_gh_cpu_<version>
```

Shipped outputs: `kaggle_out/chenyuefei_notebookf4aab87efb/plan/reports_gh_cpu_2.10/` (side A, 2.10) and
`kaggle_out/chenyuefei_notebook955c6cac50/plan/reports_gh_cpu_2.14/` (side B, 2.14). The first table of each
`KNOWN_BUGS.md` gives the detections per issue state: 2.10 closed 50/123, open 32/70 (82/193); 2.14 closed 16/123,
open 29/70 (45/193). The per-program table below it lists the detected programs only. The paired outcomes
(both 41, older only 41 = 35 closed + 6 open, newer only 4, neither 107) follow by joining the two lists:

```
python -c "import re,collections as C;L=lambda p:dict(re.findall(r'\| (issue_\S+) \| #\d+ \| (open|closed) \| yes \|',open(p,encoding='utf-8').read()));a=L('kaggle_out/chenyuefei_notebookf4aab87efb/plan/reports_gh_cpu_2.10/KNOWN_BUGS.md');b=L('kaggle_out/chenyuefei_notebook955c6cac50/plan/reports_gh_cpu_2.14/KNOWN_BUGS.md');print('both',C.Counter(a[k] for k in a if k in b),'older',C.Counter(a[k] for k in a if k not in b),'newer',C.Counter(b[k] for k in b if k not in a))"
```

"Neither" = 70 open / 123 closed minus the other three cells. The audit of the 35 closed older-only cases is
`results/p02_audit/historical_pairs.csv` (Section 11.2): column `fix_in_new_version` = `yes` for 19 rows (repair
commit included in v2.14.0), `probable: in v2.14.0` for 1, `unknown` for 15. Repair commits can be re-checked
read-only with

```
python scripts/fix_in_release.py v2.14.0 <issue> [<issue> ...]
python scripts/linked_prs.py v2.14.0 <issue> [<issue> ...]
```

(60 GitHub requests per hour without a `GITHUB_TOKEN`). `scripts/build_benchmark.py --old ... --new ...
--verified` turns two such report directories into `benchmark/historical_bugs.json`. The motivating example of
Section 2.3 (issue #195422) is one of these scripts:
`reproducers_gh/issue_195422_inductor_torch_compile_silently_drops_an.py`, detected on 2.10 and not on 2.14
(`benchmark/historical_bugs.json`, entry PT-0035).

---

## 10. Section 4.6: Discussion

### 10.1 Supplementary exploration

* **89 later program additions, 274-program corpus.** `tcc/dynamo_semantics_batch4.py` (57 programs) and
  `tcc/dynamo_semantics_batch5.py` (33 programs, exception routing; one name collides with batch 3, hence 89 new
  programs). Commands: `python run.py dynsem --out <dir>` (plain eager-vs-compiled; `--dynamic`, `--only`) and
  `python scripts/graph_break_insertion_diff.py --backend eager` (also `aot_eager`, `inductor`, `--dynamic`,
  `--relation state_marker|disable_call|print_marker`). Results: `results/graph_break_insertion/<variant>/report.json`
  (the shipped `eager/report.json` covers 274 programs), logs `results/graph_break_insertion_eager_b4.log`,
  `_b5.log`, `_aot_eager_b4.log`; C44/C45 reproducers are sub-cases of `scripts/oracle_tiers_cases.py`.
* **CUDA dtype sweep (383 divergent pairs, 73 CUDA-only).** Kaggle job `edgeunsupgpu`
  (`opinfo_edge_sweep.py --unsupported --device cuda`, T4). Output `kaggle_out/edgeunsupgpu/cases.jsonl` (1,303
  (op, dtype) pairs). Pairs with a `SILENT` diff: 383; 73 of them are not silent in `kaggle_out/edgeunsup/cases.jsonl`
  (CPU), 310 are shared, 96 are CPU-only.
* **Boundary and precision probes (erf on very small float32 values).** `python scripts/special_vs_mpmath.py`
  (no options; writes `results/special_mpmath.jsonl`) and `python scripts/binary_vs_mpmath.py`
  (`results/binary_mpmath.jsonl`); singular-value boundary sweep with `scripts/opinfo_edge_sweep.py`.

### 10.2 Table 11: negative evidence

| Sweep | Command | Source and reading |
|---|---|---|
| Disk-cache reuse (29 cases) | `python run.py diskcache --out results_repro/diskcache` (`--cases a,b`, `--timeout`) | `results/diskcache/DISKCACHE.md` (24 rows, 763 s) and `results/diskcache_extra/DISKCACHE.md` (5 rows, 265 s); `requires_grad` appears in both (infra error in the first run, re-run in the second). Verdict column: `STALE` = 0 in both. Rows whose AOTAutograd-cache column shows a hit: `tensor_const_global`, `closure_tensor_factory` (2 reuse hits); `freezing_param` is a `hit-ok` row with an FX-graph cache hit only. Raw data: `diskcache.json`. |
| Guard changes (26 context switches) | `python run.py cache ...` (Section 8.3) | `results/rq4_cache.json` -> `real_compiler_switch_table` (26 rows, `verdict` = `none` in every row); table also in `results/RQ4.md` |
| Export values (520 operators) | `python scripts/export_roundtrip_edge.py --dtype float32` (resumable; writes `results/export_roundtrip/float32.jsonl`, about 5 s per operator) | `python scripts/export_roundtrip_report.py` prints `520 ops; {'ok': 477, 'DIFF': 21, 'NO_SAMPLES': 22}` and lists the differences; none is specific to the save/load or AOTInductor route (all also occur under plain `torch.compile`, or are known reports) except the two process crashes |
| Teardown crash C46 (Windows) | as above; minimized case `aoti_runner_destroyed_n4096` | `python scripts/oracle_tiers_real.py run --only C46` re-runs it; `slice_scatter` / `kron` CRASH rows in the export report |
| CUDA-graph outputs (19 cases) | Kaggle job `gpustale` (`scripts/cudagraph_stale_sweep.py`, needs CUDA) | `kaggle_out/gpustale/plan/results/cudagraph_stale/cases.jsonl` (19 lines; keys `case`, `stale`, `error`, `value`) |
| Higher-order operators (47 programs) | `python scripts/hop_consistency_diff.py --backends eager,aot_eager,inductor` | `results/hop/cases.jsonl` (47 lines), logs `results/hop_run.log`, `hop_run2.log` |

---

## 11. Supplementary experiments of 2026-09-25

### 11.1 P0-1: paired graph-break recomputation

Recomputes the RQ1 experiment on the frozen 185 programs in one process with one oracle and three arms, and
records static insertion positions, breaks actually hit, frames per arm, the per-call divergence kind and the
paired classification (same / new / gone / changed / same-divergence). Commands and readings: Section 6.
Outputs: `results/p01_audit/` (`audit_eager.json`, `audit_aot_eager.json`, `pairs_*.csv`, `audit_*.log`,
`minimal/`). `pairs_<backend>.csv` has one row per program and call. Running `scripts/gb_p01_summary.py` regenerates
the Markdown reports `results/p01_audit/P0-1_<backend>.md` (shipped).

### 11.2 P0-2: evidence audit

* Ledger and status check of the 73 report items: `results/p02_audit/items.csv` (one row per item: source run,
  versions and platforms, contract validity, duplicate relation, root-cause id, status bucket and its evidence) and
  the summary `results/p02_audit/AUDIT.md`. The per-item table was assembled from tracker data and internal notes
  that are not part of the artifact; the other shipped evidence is `data/findings.csv` and the read-only tracker
  dump `results/issue_status/raw.json` (refresh: `python scripts/issue_status_report.py`).
* 35 historical buggy/fixed pairs: `results/p02_audit/historical_pairs.csv`, one row per pair with the issue,
  reproducer path, result on each version, report directories, environment of both sides, issue state, fix PR or
  commit, `fix_in_new_version`, `fix_evidence`, `evidence_type` (`version-diff + fix commit` 19, `version-diff
  only` 16) and `checked_at`. The fix evidence was collected with read-only GitHub queries of the kind done by
  `scripts/fix_in_release.py` and `scripts/linked_prs.py`; the script that assembled the CSV is not included.

### 11.3 P1-1: baselines (random seeds, NNSmith, TorchProbe)

Random and NNSmith: Section 8.1 and 8.5. Full description, commands and timings:
`results/p11_baselines/README.md`.

**TorchProbe setup** (details: `results/p11_baselines/TORCHPROBE.md`):

1. The paper gives no repository; the implementation is the first author's repository
   `https://github.com/soodoshll/temisu`, commit `3aa039c4882f34400cdde9199fc39e688823d4a4`:

   ```
   git clone https://github.com/soodoshll/temisu
   cd temisu
   git checkout 3aa039c4882f34400cdde9199fc39e688823d4a4
   ```

2. It needs `nnsmith` (0.1.0 was used) and pins nothing else. Run verbatim, `python -m temisu.fuzz` stops with
   `AssertionError: Torch not compiled with CUDA enabled` on a CPU-only build (`device = 'cuda'` is hard-coded and
   the module runs an unbounded loop that also deletes `./report`).
3. One-line patch needed on Python 3.11 and later: in `temisu/mutator.py` line 46 replace
   `random.sample(tensor_map.items(), 2)` with `random.sample(list(tensor_map.items()), 2)`.
4. A bounded driver is needed: the loop body of `fuzz.py` with `device` as a parameter (`cpu`), a fixed number of
   seeds, `model_gen(..., seed=s)`, `tcc.compat.ensure_msvc_env()` on Windows, an extra eager validity check per
   mutated program, and saving of every program (`<name>.py`) with its module list and inputs (`<name>.pt`). The
   driver is `scripts/torchprobe/run_temisu.py` and the patch is `scripts/torchprobe/temisu_py311.patch`; clone
   `temisu` into `scripts/torchprobe/temisu` and apply the patch there (`git apply`) before running it. It was invoked as
   `run_temisu.py --device cpu --seeds 35 --first-seed 0 --out tp_out` (in two batches, 0-9 and 10-34) and
   `--seeds 15 --first-seed 100 --max-nodes 20 --out tp_out20` (stopped after 47 tests). Its record format is that
   of `results/p11_baselines/torchprobe_tests_n10.jsonl`; the saved program sources are in
   `torchprobe_programs_n10/`, `torchprobe_programs_n20/` (the `.pt` tensors are not included).
5. Summaries and Setting B (TorchProbe programs judged by the Tcc harness against the 19 injected faults):

   ```
   python scripts/p11_torchprobe_summary.py <tp_out>/tests.jsonl --out results_repro/p11 --tag n10_seeds0to34
   python scripts/p11_torchprobe_applicability.py --out results_repro/p11
   python scripts/p11_torchprobe_setting_b.py --programs <tp_out>/programs --temisu <temisu clone> --mutations origin --out results_repro/p11
   python scripts/p11_torchprobe_setting_b.py --programs <tp_out>/programs --temisu <temisu clone> --mutations subfunction --out results_repro/p11
   ```

   Other options: `--tests-jsonl` (keep eager-valid programs only), `--max-programs`, `--first-program`,
   `--budget`, `--backend`, `--tag`, `--report-dir`. Recorded: 583.9 s for 175 tests (`max_nodes=10`), 349.4 s for 47
   tests (`max_nodes=20`, shared CPU), Setting B 127.9 s (`origin`) and 310.1 s (`subfunction`).

Outputs: `torchprobe_run_*.json/.md` (222 tests, 0 inconsistent, 0 crash, 0 invalid),
`torchprobe_applicability*.json/.md`, `torchprobe_setting_b_origin.json` and `torchprobe_setting_b_subfunction.json`
(`_origin_a`, `_origin_b` are superseded partial runs).

### 11.4 P1-2: oracle tiers on real root causes

Section 7. Subcommands of `scripts/oracle_tiers_real.py`: `run [--only IDS] [--workers N] [--timeout S]
[--primary-only]`, `control-corpus [--workers N] [--timeout S]`, `control-autocast [--acdtype D] [--samples N]
[--shard i/n]`, `report`, `eval <trace>`. Outputs in `results/p12_oracle_tiers/`: `REPORT.md`, `root_causes.csv`
(52 root causes with member report items), `subcases.csv` (one row per sub-case, with eager and compiled exception
and exit status), `tiers.csv`, `summary.json`, `traces/`, `control_traces/`, `control_autocast_*.jsonl`,
`review_times.csv` (time from first divergence to filing, where recorded; mostly "not recorded").

### 11.5 P1-3: timing

`python scripts/gb_p01_audit.py --backend eager --timing 3 --out results_repro/p13_timing` adds three fresh runs per
arm (`torch._dynamo.reset()` before each) and records per-run wall time; then
`python scripts/gb_p01_summary.py --backend eager --out results_repro/p13_timing` renders it. Shipped:
`results/p13_timing/audit_eager.json` (`timing_baseline`, `timing_break`, per-program `timing_runs`),
`pairs_eager.csv`, `timing_eager_run1.log`. Another CPU-heavy process was active during the recorded run.

---

## 12. Result directory -> experiment

Top-level output directories:

| Directory | Experiment |
|---|---|
| `results/` | all local (Windows) experiment outputs; sub-directories listed below |
| `results0909/` | earlier download (2026-09-09) of the Kaggle T4 campaign outputs (`plan/gpu_reports_corpus`, `gpu_reports_gh`, `gpu_reports_gh_cpu_set`, `gpu_reports_opinfo`) produced by `kaggle/campaign_gpu.sh`, torch 2.10.0+cu128 |
| `kaggle_out/` | outputs of the Kaggle jobs, one directory or log per job name of `kaggle/drive.py` (`<job>/`, `<job>.log`, `_kernel_<job>/` kernel metadata, `_wait_*.txt` poll logs); `chenyuefei_notebook*/` are the two historical-replay notebooks (Table 10) |
| `gpu_reports_corpus/` | Kaggle T4 campaign over the corpus plus 10 NNSmith models (`run.py campaign --nnsmith 10`), 39 programs, 0 candidates |
| `gpu_reports_gh/` | Kaggle T4 campaign over 336 mined reproducers including CUDA ones (`reproducers_gh_cuda/`) |
| `gpu_reports_gh_cpu_set/` | Kaggle T4 re-run of the 193 CPU reproducers (`reproducers_gh/`), torch 2.10.0+cu128; first version of the historical benchmark |
| `reports/` | first real-compiler campaign: 35 corpus programs (`results/rq5_campaign.log`), 0 candidates |
| `reports_aoti/` | export / AOTInductor reproducers (`return_types_packaging`, `lifted_constant_mutation`) from `run.py aoti` |
| `reports_campaign2/` | second campaign: corpus plus 20 NNSmith models, `--keep-info` (`results/rq5_campaign2.log`) |
| `reports_decomp/` | minimized reproducers from the decomposition and meta differential (`run.py decomp`) |
| `reports_dynsem/` | reproducer of the Dynamo `random` module finding (`run.py dynsem`) |
| `reports_fx/` | torch.fx code-generation and pickle round-trip reproducers (`scripts/fx_codegen_vs_interpreter.py`) |
| `reports_gh/` | local (Windows, 2.14.0+cpu) campaign over the mined reproducers, with `KNOWN_BUGS.md` from `scripts/gh_summary.py` |
| `reports_gh_rerun/` | re-run of 13 reproducers (`reproducers_gh_rerun/`) |
| `reports_intub/` | integer / cast / IEEE special-value reproducers (`scripts/int_ub_sweep.py`) |
| `reports_layout/` | layout and alias reproducers incl. the adaptive_max_pool3d gradient case (`scripts/layout_alias_sweep.py`) |
| `reports_metamorphic/` | reproducers from the metamorphic relations between compiled runs (`run.py metamorphic`) |
| `reports_models/` | local campaign over torchvision-model reproducers (`reproducers_models/`, `scripts/make_model_reproducers.py`) |
| `reports_opinfo/` | campaign over 120 OpInfo samples (`results/rq5_opinfo.log`) |
| `reports_opinfo_all/` | campaign over 545 OpInfo samples (`results/rq5_opinfo_all.log`) |
| `reports_opinfo_s3/` | campaign over 540 OpInfo samples, seed 3 (`results/rq5_opinfo_s3.log`) |
| `reports_opinfo_s4x3/` | campaign with 3 samples per operator, 1,393 programs, 20,713 tests (`results/rq5_opinfo_s4x3.log`) |
| `reports_pairs/` | stand-alone reproducers for operator-pair and dtype families (unsupported dtypes, conjugate views under dispatch modes, half-precision intermediates, special functions) |
| `reports_rewrites/` | rewrite-precondition reproducers incl. the MSVC cases (`scripts/rewrite_precondition_diff.py`, `scripts/cxx_intrinsic_algebra_diff.py`) |
| `reports_scalar_seq/` | stale Python-float argument reproducers (`scripts/scalar_arg_sequence_diff.py`) |
| `reports_sideeffects/` | side-effect and closure-mutation reproducers (`scripts/side_effect_diff.py`) |
| `reports_smoke/` | first Inductor smoke run (`results/run_smoke.log`) |
| `reports_surgery/` | model-surgery and safety-mechanism reproducers (`scripts/model_surgery_diff.py`) |
| `reports_symint/` | SymInt / SymFloat arithmetic reproducers (`scripts/symint_arith_diff.py`) |
| `reports_symshape/` | symbolic-size indexing reproducers and the 32-bit indexing check (`scripts/symshape_index_diff.py`) |
| `reports_validation/` | lost-validation reproducers (`scripts/error_parity_sweep.py`, `scripts/error_parity_batch2.py`) |
| `reports_xtarget/` | Numba / JAX / torch-NumPy reproducers (`scripts/xtarget_numpy_diff.py`) |

Sub-directories and files of `results/` that feed the paper:

| Path in `results/` | Experiment |
|---|---|
| `p01_audit/` | P0-1, Table 4 |
| `p13_timing/` | P1-3, Table 4 timing rows |
| `p12_oracle_tiers/` | P1-2, Table 5 |
| `p11_baselines/` | P1-1, Tables 6 and 8 |
| `p02_audit/` | P0-2, Table 10 audit |
| `rq1_benchmark.json`, `rq1_bench.log`, `bench_reports/` | full Tcc on injected faults (layer diagnosis, Table A1 #P); sample issue drafts |
| `rq2_baselines.json`, `RQ2.md`, `rq2_baselines.log` | Table 6, Table 8 |
| `rq3_ablation.json`, `rq3_ablation.log` | Figure 5 |
| `rq4_cache.json`, `RQ4.md`, `rq4_cache.log` | Table 7, Table 11 guard row |
| `graph_break_insertion/`, `graph_break_insertion_*.log` | RQ1 discovery runs and Section 4.6 corpus extensions |
| `diskcache/`, `diskcache_extra/` | Table 11 disk-cache row |
| `export_roundtrip/`, `export_roundtrip_float32.log` | Table 11 export row, C46 |
| `hop/`, `hop_run*.log` | Table 11 higher-order operators |
| `autocast/`, `autocast_*.log` | autocast sweep (negative result; source of the 924 autocast controls) |
| `xtarget/`, `ort/`, `cxx_algebra/`, `host_opt/` | Section 4.5 other targets |
| `special_mpmath.jsonl`, `binary_mpmath.jsonl` | Section 4.6 precision probes |
| `pairs/` | dispatch-mode sweeps and the unsupported-dtype family list |
| `issue_status/` | read-only tracker dump (2026-09-25) |
| `selfcheck_torch.log` | Section 3 |

The remaining `results/` entries are exploratory sweeps named after the command or script that wrote them, for
example `decomp_*` (`run.py decomp`), `meta_*` (`run.py metamorphic`), `binding_*` (`run.py binding`),
`aoti_full` (`run.py aoti`), `dynsem*` (`run.py dynsem`), `dtypes_aot` (`run.py dtypes`), `complex_grad`,
`compiled_autograd`, `forward_ad*`, `vmap`, `optim`, `large_reduce`, `reuse_env`, `unary_dtype`, `layout_alias`,
`int_ub`, `side_effects*`, `surgery`, `symint`, `symshape`, `scalar_seq`, `rewrites`, `reconstruct`, `fx_codegen`,
`custom_op`, `cppwrap`, `error_parity`, `opinfo_edge`, `export_edge`, `prefix` (each from the script of the same
stem in `scripts/`), `kaggle_*.log` (Kaggle polling logs), `dedup_*` (duplicate searches), `_*_smoke` (smoke tests),
`experiments.json` / `EXPERIMENTS.md` (design experiments A-E, `run.py experiments`) and `metrics*`
(`run.py metrics`). They are not used for numbers in the paper beyond the ones listed above.

---

## 13. What cannot be reproduced locally; re-running the Kaggle jobs

**Not reproducible on a Windows CPU machine:**

* CUDA / Triton items: B5 (Table 5 excludes it), the CUDA dtype sweep (383 / 73 pairs), the CUDA-graph sweep
  (19 cases), all T4 re-checks. They need a CUDA GPU (a Tesla T4 was used).
* Linux CPU items: the historical replay of Table 10 (2.10 vs 2.14 on the same Linux image), the 1,345-pair C37 sweep,
  `gcc` / `clang` variants of the C++ differential.
* Windows-only behavior: the teardown crash C46 (immediate destruction of a loaded AOTInductor runner) and the
  `/O1`, `/O2` MSVC folds reproduce only on Windows with MSVC; Linux runs do not show them.
* Older or development builds: torch 2.10 and nightly 2.15.0.dev wheels of the dates used may no longer be
  downloadable. Results on newer builds can differ because defects get fixed.
* Report status (Table 9 status columns, Section 4.5 counts): these depend on the tracker state at a given date and
  change over time. They are reported as of 2026-10-01 and cannot be recomputed from shipped files.
* Wall times depend on the machine; the recorded runs shared the CPU with another process.

**Re-running Kaggle jobs.** Requirements: a Kaggle account with phone verification (for GPU and internet access),
the `kaggle` command-line client (`pip install kaggle`), and your own API credentials (Kaggle -> Settings -> API ->
Create New Token, saved as `%USERPROFILE%\.kaggle\kaggle.json`, or configured with `kaggle config`). Never put a
token into the artifact or into job files.

```
python kaggle/pack.py                          # writes ..\plan_kaggle.zip (entries under plan/)
python kaggle/drive.py dataset                 # create or update dataset <your user>/tcc-plan from that zip
python kaggle/drive.py push bench210 bench214  # push and start jobs (script kernels named tcc-<job>)
python kaggle/drive.py status bench214         # running / complete / error
python kaggle/drive.py log bench214            # last lines of the job log
python kaggle/drive.py pull bench214           # download the output into kaggle_out/bench214/
python kaggle/drive.py wait bench210 bench214  # poll every 5 minutes, pull when complete
```

The job names are the keys of `JOBS` in `kaggle/drive.py` (one script per job in `kaggle/jobs/`; the `needs_gpu`
flag requests a Tesla T4). Each job copies the dataset to `/kaggle/working/plan`, installs the torch version it
needs, runs its commands and leaves outputs under `/kaggle/working`. Jobs used in the paper: `bench210`, `bench214`
(Table 10), `edgeunsup`, `edgeunsupgpu` (C37 and the CUDA sweep), `gpustale` (CUDA graphs), `dynsemlinux` (Linux /
Python 3.12 / nightly re-check of the Dynamo corpus and graph-break runs), `xtargetlinux`, `cxxalgebra`,
`nightlycpu`, `nightlygpu`, `gpuverify` and the `*linux` / `*gpu` re-check jobs of individual reports. Some job
files are generated by `kaggle/make_*.py` (for example `make_opinfo_edge_job.py`, `make_recheck_*.py`). A kernel log
can be fetched without the output files with `python kaggle/kernel_log.py <owner/slug> <out.log>` and printed with
`python scripts/kaggle_log.py <file.log> [regex]`. The interactive notebook route (upload the zip as a dataset, run
`kaggle/bootstrap.sh` or `kaggle/campaign_gpu.sh` in a T4 notebook) produces the same outputs.

---

## 14. Known gaps in this artifact

These numbers or files could not be traced to a shipped producing command, or differ from the shipped data:

* **Table 9 status columns and the "13 known defects"** (Section 9.1): the 2026-10-01 classification and the list of
  known defects are not stored in shipped files. `data/findings.csv` holds the 2026-09-23 snapshot.
* **C37 "418 differences"** (Section 9.3): `kaggle_out/edgeunsup/cases.jsonl` has 1,345 pairs and 134 operators as in
  the paper, but 443 rows with verdict `DIFF:SILENT`; the filter that gives 418 is not recorded.
* **Disk-cache "two reuse hits"** (Section 10.2): the verdict column lists three `hit-ok` rows; two of them hit both
  caches, one (`freezing_param`) only the FX-graph cache. The paper's count matches the AOTAutograd-cache column.
* **TorchProbe tensors**: the `.pt` input files saved by `scripts/torchprobe/run_temisu.py` are not included; rerunning the driver regenerates them (Section 11.3, step 4).
* **`results/p02_audit/historical_pairs.csv`** was assembled by a script that is not included (Section 11.2); the
  underlying queries can be repeated with `scripts/fix_in_release.py` and `scripts/linked_prs.py`.
* **B7 primary sub-case** of the oracle-tier analysis reads an issue draft that is not included (Section 7).
* Issue drafts under `0913issues/` and `0920issues/`, referenced from some shipped Markdown files, are not part of
  the artifact; the numbers they contained are reproduced from the JSON and CSV files named above.
