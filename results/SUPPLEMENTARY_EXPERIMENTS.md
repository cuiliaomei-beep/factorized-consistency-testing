# Supplementary experiments (2026-09-25)

These experiments add comparability and evidence to the main study without new large-scale sweeps. Every number below is recomputed by the scripts and data files listed in each section. Quantities that were not measured are marked "not recorded".

Machine: Windows 11, Python 3.14.7, torch 2.14.0+cpu, no GPU. GitHub was queried read-only.

| Item | Status | Outputs |
|---|---|---|
| P0-1 Recomputation of the graph-break experiment | done | `results/p01_audit/` (`P0-1_eager.md`, `P0-1_aot_eager.md`, `audit_*.json`, `pairs_*.csv`, `minimal/`) |
| P0-2 Evidence audit | done | `results/p02_audit/` (`AUDIT.md`, `items.csv`, `historical_pairs.csv`) |
| P1-1 Comparison with the closest methods | done | `results/p11_baselines/` (`TORCHPROBE.md`, `random_seeds.md/.csv`, `nnsmith_seeds.md`) |
| P1-2 Oracle analysis on real defects | done | `results/p12_oracle_tiers/` (`REPORT.md`, `root_causes.csv`, `tiers.csv`, `traces/`) |
| P1-3 Cost | done | `results/p13_timing/P1-3.md` |
| P2 Transfer to a second system | not done | the full evaluation is limited to PyTorch |

Scripts: `scripts/gb_p01_audit.py` and `scripts/gb_p01_summary.py` (P0-1, P1-3); `scripts/oracle_tiers_cases.py` and `scripts/oracle_tiers_real.py` (P1-2); `scripts/p11_*.py` and the commands in `results/p11_baselines/README.md` (P1-1); `scripts/issue_status_report.py` (issue status, read-only).

## P0-1 Graph-break experiment on the frozen 185-program corpus

The corpus is the first three batches of `tcc/dynamo_semantics*.py`: 77 + 74 + 34 = 185 programs. One program of batch 4 has the same name as a batch-3 program (`chained_comparison_tensor_scalar`) and replaces it on import, which is why batches 1-4 contain 241 rather than 242 programs. The audit script reloads batch 3 to restore the frozen version.

| Quantity | Value |
|---|---|
| Runnable / rewritable programs | 185 / 185 |
| Calls per arm | 370 (2 per program); 1110 over the three arms (eager, compiled original, compiled break variant) |
| Static insertion positions | 477 (11 programs have none) |
| User graph breaks actually hit (Dynamo counter, both calls) | 544; all breaks hit in 155 programs, fewer than inserted in 19 |
| Graph segments (`frames.total`) | original arm 614, break arm 1175 |
| Divergence instances, original / break / intersection | 19 / 25 / 17 |
| Paired classification | same 343, same divergence 13, new 8, gone 2, changed 4 |
| Independent root causes among the new divergences | 2: C41 (`collections_types`) and C42 (`gen_send`, `gen_throw`, `gen_return_value_stopiteration`) |
| Expected behavior / invalid | 0 (the `lru_cache` inlining concerns batch-4 programs, outside the frozen corpus) |
| Gone (2) | `python_random`: the baseline divergence is the known item B17; with the breaks the compiled values equal eager |
| Changed (4) | `tensor_subclass_torch_function`, `threading_local_state`: baselines that already diverge, with a different state log |
| Does the minimal reproducer still need a break? | C41: no. C42: the "generator returned" form needs none; the "send/throw while suspended" form uses one explicit break as the suspension point |

The counts agree item by item with the original run of 2026-09-22. The aot_eager arm also has 8 new divergences and 2 root causes; its baseline is 21 instead of the 20 recorded on 2026-09-22 because `deepcopy_module_state` is unstable between runs. "477" is the number of static insertion positions; 544 breaks were actually hit.

## P0-2 Evidence audit

**73 reported items** (`results/p02_audit/AUDIT.md`, Part A)

| Quantity | Value |
|---|---|
| Report items | 73 = 55 new issues + 15 comments + 3 reports to the MSVC vendor |
| Candidates | 102 numbered ledger rows (72 cited by the 73 items; 30 not cited: 21 negative results, 2 declined, 7 other) + 22 candidates rejected before numbering |
| Independent root causes | 64; six root causes cover several items (RC01 = B1/C32/C33/C39, RC30 = C3/C26, RC49 = C22/C31, RC56 = C30/C37/C38, RC62 = C42/C43, RC63 = C44/C45); the 55 new issues map to 54 root causes |
| Contract validity | valid 66; not applicable (other compiler) 4; invalid 2 (B16 crashes on both paths, C12 hangs on both paths); expected behavior 1 (B13) |
| Corpus sizes | 185 / 241 / 274 are used consistently; one sentence of the paper draft says "90 programs added afterwards" where there are 89 distinct programs (90 decorators, one duplicate name) |

Status of the 73 items (four buckets: Fixed = a fix merged or a developer opened a fix PR; Confirmed = a maintainer confirmed or someone reproduced / root-caused it; Pending = no human response; Rejected = judged expected behavior):

| Snapshot | Items | Fixed | Confirmed | Pending | Rejected |
|---|---|---|---|---|---|
| 2026-09-23 (used in `data/findings.csv`) | 73 | 14 | 10 | 48 | 1 |
| 2026-09-25 re-check (`results/p02_audit/items.csv`) | 73 | 15 | 14 | 43 | 1 |

Seven items moved between the two dates, all through GitHub activity after 2026-09-23: C8 #197893 Confirmed to Fixed (fix PR #198446); C41 #198189 Pending to Fixed (PR #198518); B22 #197106, C29 #198081 and C42 #198190 Pending to Confirmed; C37 #198155 Pending to Confirmed (a draft PR covers 6 of the 134 operators and a maintainer questions the restriction); B16 #197099 Fixed to Confirmed (its only fix PR #197152 was closed unmerged on 2026-09-24). If a closed, unmerged fix PR still counts as "a developer opened a fix PR", B16 stays Fixed and the totals are 16 / 13 / 43 / 1. Issue status changes over time; re-run `scripts/issue_status_report.py` for the current state.

**35 historical buggy/fixed pairs** (Part B)

| Quantity | Value |
|---|---|
| Same device and platform | 35 / 35 (Kaggle Linux CPU, torch 2.10.0+cpu vs 2.14.0+cpu, same job package) |
| Detection rates behind RQ5 | closed issues 50/123 = 40.7% to 16/123 = 13.0%; open issues 32/70 = 45.7% to 29/70 = 41.4% |
| Fix commit known to be in v2.14.0 | 19 (12 from the closing commit + 7 found by commit search), 1 probable, 15 version difference only |
| Divergence kind on 2.10 | exception 15, value 14, metadata + value 3, alias + mutation 1, unknown 2 |
| Result on 2.14 | 35 / 35 consistent, no harness error |
| Caveats | #182399 and #181870 share one fix PR; #178125 and #178128 share one closing commit and their fix PR landed after 2.14; the 2.14 side did not record `environment.txt` |

## P1-1 Comparison with the closest methods

**TorchProbe** (`results/p11_baselines/TORCHPROBE.md`). The paper gives no repository; the implementation is the `temisu` repository of the first author (commit `3aa039c4`, 2023-03-20, 1087 lines). Run verbatim it fails because `device='cuda'` is hard-coded. Two portability changes make it run on Python 3.14 / torch 2.14 / nnsmith 0.1.0: a driver that makes the device a parameter and bounds the number of seeds, and a one-line change in `mutator.py` (`random.sample(dict.items())` to `random.sample(list(...))`, required on Python 3.11 and later).

| Quantity | Value |
|---|---|
| Applicable programs | 3 of the 29 controlled-benchmark programs can be expressed as TorchProbe seeds (blockers: control flow 16, Python reads of tensor metadata 11, operators without a renderer, ...); 0 of the 185 Python-corpus programs |
| Own seeds | `max_nodes=10`: 35 seeds, 175 tests; `max_nodes=20`: 10 seeds, 47 tests. All pass: 0 inconsistencies, 0 crashes, 0 invalid tests |
| Wall time | 175 tests in 583.9 s (median 2.79 s per test, first compilation 18.7 s); 47 tests in 349.4 s (CPU shared with P1-2) |
| Correspondence with reported defects | none; no new findings |
| Setting B (TorchProbe's 35 programs judged by our harness against the 19 injected faults) | 5/19 unmutated, 6/19 after the four mutations; 0 false positives on the unmodified compiler |

The harness's raw Setting-B counts (6/19 and 15/19) include two artifacts. First, the injected-fault wrapper `_make_scalar_boundary` evaluates `if tensor` on foreign programs whose argument is a tensor and raises; the same artifact is in the NNSmith leg, which is 5/19 without it. Second, after several hundred compile/reset cycles in one process, Dynamo raises `dictionary changed size during iteration` (a race between `get_items_from_dict` iterating a live `globals().items()` view and garbage-collection callbacks on Python 3.14; fixed upstream by PR #191281 on 2026-08-20, not in v2.14.0; not reproducible standalone).

Because TorchProbe applies to none of the Python corpus, it cannot be ranked against the method on that corpus.

**Random baseline per seed** (`results/p11_baselines/random_seeds.md`; budget 12 tests per program, value oracle, the B1 setting of RQ2)

| Seeds | Faults detected (of 19) | Tests | Invalid | Wall time (s) |
|---|---|---|---|---|
| 0 / 1 / 2 | 11 / 12 / 11 | 458 / 483 / 457 | 109 / 108 / 120 | 45.3 / 55.7 / 49.4 |
| 3-9 | 11, 11, 11, 6, 10, 11, 10 | 486-559 | 75-126 | 47.2-79.4 |
| all 10 | mean 10.4, median 11, min 6, max 12, sample sd 1.65 | mean 492.7 | mean 107.5 (17.9%) | mean 54.1 |

The paper's 11.3 is the mean of seeds 0-2. Per-fault detection frequency and time to first detection are in `random_seeds_per_fault.csv`.

**NNSmith (B3, native input, 3 model seeds)**: 6 / 6 / 6 faults (mean 6.0, sd 0), 8 models and 152 tests per seed, 0 invalid; 5/19 without the wrapper artifact. The 9 reported for NNSmith in the paper is the leg with our contexts added.

## P1-2 Oracle analysis on independent real root causes

Unit: 73 items - 5 other-compiler items - 10 items outside the compile pipeline = 58 items; merging comments into their issues and items with one root cause gives **52 independent root causes** with 127 sub-cases (the triggering programs listed in each report). Each sub-case runs once in eager and once compiled (child processes, fresh caches); all observables go into one trace, and three cumulative oracle tiers are evaluated offline on the same trace, reusing `tcc.oracle` and `tcc.observe`.

| | T1 values + ordinary crash/exception | T2 + metadata | T3 + alias/state/gradient/exception routing/multi-precision/process | not reproducible here |
|---|---|---|---|---|
| all sub-cases (51 root causes reproduced) | 36 | 38 | 51 | 1 (B5 needs CUDA) |
| primary sub-case only | 34 | 38 | 51 | |

Weakest detecting tier: T1 36, T2 2 (B15 dtype, B25 stride), T3 13 (B12, C5, C22, C23, C37: exception routing, the lost-validation family; C1 reverse exception routing; B18 exception type; C8 aliasing; C11 and C41 side effects; C34 gradient; C40 multi-precision reference; C46 process crash at runner destruction).

False positives with the same tracer and tiers on consistent control sets: the 29 unmodified corpus programs give 0 alarms in every tier; the autocast negative-result set (462 operators x 2 dtypes = 924 samples) gives 2 alarms in every tier (`bilinear` and `linalg.polar` under bf16, differences of a few ULPs that the paper's 3e-2 tolerance does not report). T2 and T3 add no false positives over T1.

Manual review time: clock times exist only for 2026-09-22, giving upper bounds (C41/C42 sweep to draft at most 15 min, draft to report at most 19 min; C44 at most 20 min; C40 at most 1 h 45 min; C46 at most 1 h 14 min; C37 50 min plus at most 45 min). Other items: not recorded.

Harness observations: the floating-point comparison in `tcc/oracle.py` masks all differences when the eager output contains a NaN (C13 is reported in T1 only through a position-wise NaN check); `_exact_equal` treats an int/bool dtype difference as a value difference; B7 depends on a jvp having run earlier in the process.

## P1-3 Cost (frozen 185 programs, 3 fresh runs per arm)

| Arm | Valid tests (calls) | Compiled frames | Total wall time per run (s) | Median | Min-max |
|---|---|---|---|---|---|
| original | 370 | 614 | 13.59, 13.23, 13.09 | 13.23 | 13.09-13.59 |
| break variant | 370 | 1175 | 32.20, 32.65, 32.97 | 32.65 | 32.20-32.97 |

The break arm compiles 1.9 times as many frames and takes 2.5 times the wall time. The first new root cause appears after 0.6 s of accumulated run time (`gen_send`), the second after 5.2 s (`collections_types`). The spread over the three runs is below 4%. The data support "within minutes" (median of both arms together 45.9 s) but not an equal-time efficiency claim.
