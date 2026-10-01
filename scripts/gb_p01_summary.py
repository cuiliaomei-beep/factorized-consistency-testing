"""Render results/p01_audit/audit_<backend>.json as the P0-1 Markdown report (P0-1.md) with the counts the FSE
checklist asks for, in the checklist's units: programs / calls / divergence instances / independent root causes.
Root-cause assignment of the non-'same' pairs is given explicitly below (reviewed by hand against the ledger).

    python scripts/gb_p01_summary.py [--backend eager]
"""
import argparse
import json
import os

# program -> (root cause id, verdict, explanation).  Verdicts: defect | known-defect | expected | noise
ROOT_CAUSE = {
    "collections_types": ("C41", "defect", "OrderedDict.move_to_end on a dict that becomes an input of the resume function is not replayed (#198189)"),
    "gen_send": ("C42", "defect", "generator alive across the break is reconstructed as tuple_iterator; .send raises AttributeError (#198190)"),
    "gen_throw": ("C42", "defect", "same reconstruction; .throw raises AttributeError (#198190)"),
    "gen_return_value_stopiteration": ("C42", "defect", "same reconstruction; StopIteration.value becomes None (#198190)"),
    "python_random": ("known-B17", "known-defect", "baseline diverges on both calls (values differ from eager) = the known item B17, random.seed inside a compiled function ignored (fixed upstream); with a break after every statement seed and draws are no longer in one traced frame and the compiled values equal eager, so the divergence disappears"),
    "tensor_subclass_torch_function": ("known-state", "known-defect", "baseline already diverges in the __torch_function__ call log; the break changes which calls are logged, not whether it diverges"),
    "numpy_scalar_types": ("known-numpy", "known-defect", "aot_eager arm only: baseline already diverges (numpy scalar promotion, known family C13/C20); under the break the compiled dtype changes from float32 to float64, still a divergence"),
    "deepcopy_module_state": ("known-deepcopy", "known-defect", "aot_eager arm only: copy.deepcopy(module) inside the compiled function; the 09-22 run had call0 as 'new', this run has it diverging in both arms (baseline 21 instead of 20), so it is a baseline item, not a break finding"),
    "threading_local_state": ("known-state", "known-defect", "baseline already diverges on the threading.local object stored in STATE; the break changes the repr of the object, not the divergence"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="eager")
    ap.add_argument("--out", default=os.path.join("results", "p01_audit"))
    a = ap.parse_args()
    d = json.load(open(os.path.join(a.out, f"audit_{a.backend}.json"), encoding="utf-8"))
    s, rows = d["summary"], d["programs"]
    by = {r["program"]: r for r in rows}
    non_same = {}
    for r in rows:
        for c in r["calls"]:
            if c["pair"] not in ("same", "same-divergence"):
                non_same.setdefault(r["program"], []).append(c)
    L = []
    L.append(f"# P0-1 recomputation: frozen 185-program corpus x graph break after every statement (backend={s['backend']}, torch {s['torch']}, {s['date']})\n")
    L.append("One process, one oracle (exact comparison of return values + STATE snapshot + exception type), three arms: eager, compiled original program, compiled break variant. Every program is called twice.\n")
    L.append("## Counts (numerator and denominator reported separately)\n")
    L.append("| Quantity | Value |\n|---|---|")
    L.append(f"| Frozen programs | {s['programs']} ({', '.join(f'{k.split('.')[-1]} {v}' for k, v in s['programs_per_module'].items())}) |")
    L.append(f"| Runnable / rewritable programs | {s['programs_rewritten']} / {s['programs']} (0 AST rewrite failures) |")
    L.append(f"| Calls per arm | {s['calls_per_arm']} ({s['programs']} programs x 2 calls); {s['executions_total']} over the three arms |")
    L.append(f"| Static insertion positions (after top-level statements) | {s['static_insertion_positions']}; {len(s['programs_with_zero_static_positions'])} programs have no insertion position |")
    L.append(f"| User graph breaks actually hit (Dynamo counter `graph_break`, both calls; can exceed the static count when a frame is retraced and fall below it when a statement is not reached) | {s['user_breaks_actually_hit_total']} |")
    L.append(f"| Programs in which every static break was hit | {s['programs_where_all_static_breaks_hit']}; {len(s['programs_where_fewer_breaks_hit'])} programs hit fewer breaks than inserted |")
    L.append(f"| Graph segments (Dynamo `frames.total`, both calls) | original arm {s['frames_total_baseline']}, break arm {s['frames_total_break']} |")
    L.append(f"| Divergence instances (program x call) | original arm {s['baseline_divergent_calls']}, break arm {s['break_divergent_calls']}, intersection {s['intersection_calls']} |")
    p = s["pairs"]
    L.append(f"| Paired classification | same (no divergence) {p.get('same', 0)}; same divergence {p.get('same-divergence', 0)}; new {p.get('new', 0)}; gone {p.get('gone', 0)}; changed {p.get('changed', 0)} |")
    L.append(f"| Wall time of this run | {s['wall_s_total']} s (eager arm {s['wall_s_per_arm_sum']['eager']} s, original arm {s['wall_s_per_arm_sum']['baseline']} s, break arm {s['wall_s_per_arm_sum']['break']} s, compilation included) |")
    L.append("")
    L.append("## Pairs that are not 'same' (program x call)\n")
    L.append("| Program | Call | Pair | Original arm | Break arm | Static breaks | Breaks hit | Segments orig/break | Root cause | Verdict |\n|---|---|---|---|---|---|---|---|---|---|")
    for prog, calls in non_same.items():
        r = by[prog]
        rc = ROOT_CAUSE.get(prog, ("?", "?", "not classified"))
        for c in calls:
            L.append(f"| `{prog}` | {c['call']} | {c['pair']} | {c['baseline_kind'] or '-'} | {c['break_kind'] or '-'} | {r['static_breaks']} | "
                     f"{r['break_variant']['user_inserted_breaks']} | {r['baseline']['frames_total']} / {r['break_variant']['frames_total']} | {rc[0]} | {rc[1]} |")
    L.append("")
    L.append("## Independent root causes (new divergences after deduplication)\n")
    new_progs = [pg for pg, cs in non_same.items() if any(c["pair"] == "new" for c in cs)]
    rcs = {}
    for pg in new_progs:
        rcs.setdefault(ROOT_CAUSE[pg][0], []).append(pg)
    L.append(f"{p.get('new', 0)} new divergence instances in {len(new_progs)} programs, **{len(rcs)} independent root causes** after deduplication:\n")
    for k, v in rcs.items():
        L.append(f"- **{k}**: {', '.join('`' + x + '`' for x in v)}: {ROOT_CAUSE[v[0]][2]}")
    L.append("")
    L.append("Gone and changed pairs are not new findings:")
    for pg, cs in non_same.items():
        if all(c["pair"] in ("gone", "changed") for c in cs):
            L.append(f"- `{pg}` ({cs[0]['pair']}): {ROOT_CAUSE[pg][2]}")
    L.append("")
    L.append("Expected behavior / invalid items: none among the new divergences of the 185 programs (the `functools.lru_cache` inlining belongs to batch-4 programs, which are not in the frozen corpus).\n")
    L.append("## Breaks and graph segments of the representative programs (Dynamo counters)\n")
    L.append("| Program | Static breaks | Breaks hit | Segments, original arm | Segments, break arm | Other break reasons (break arm) |\n|---|---|---|---|---|---|")
    for pg in list(non_same):
        r = by[pg]
        L.append(f"| `{pg}` | {r['static_breaks']} | {r['break_variant']['user_inserted_breaks']} | {r['baseline']['frames_total']} | {r['break_variant']['frames_total']} | {'; '.join(r['break_variant']['other_break_reasons'])[:160] or '-'} |")
    L.append("")
    if s.get("programs_where_fewer_breaks_hit"):
        L.append("Programs that hit fewer breaks than inserted (the break follows a statement that is not reached, or the program raises before it):\n")
        for pg, st, hit in s["programs_where_fewer_breaks_hit"]:
            L.append(f"- `{pg}`: inserted {st}, hit {hit}")
        L.append("")
    if "timing_baseline" in s:
        L.append("## Paired timing (P1-3; %d additional fresh runs per arm, each compiles after `torch._dynamo.reset()` and makes two calls)\n" % s["timing_baseline"]["runs"])
        L.append("| Arm | Total wall time per run (s) | Median | Min | Max | Sum of per-program medians |\n|---|---|---|---|---|---|")
        for arm, lab in (("baseline", "original"), ("break", "break variant")):
            t = s[f"timing_{arm}"]
            L.append(f"| {lab} | {', '.join(str(x) for x in t['total_per_run_s'])} | {t['total_median_s']} | {t['total_min_s']} | {t['total_max_s']} | {t['per_program_median_sum_s']} |")
        L.append("")
    L.append("## Do the minimal reproducers still need a graph break?\n")
    L.append("See `results/p01_audit/minimal/`. The C41 reproducer contains no `graph_break` and shows different OrderedDict orders in eager and compiled execution. "
             "The C42 reproducer has two parts: the first (a generator returned from the compiled function) reproduces `tuple_iterator` and `StopIteration.value=None` without any break; "
             "the second (`.send` on a suspended generator) uses one explicit `graph_break()` as the suspension point. The return path of C41 and C42 therefore needs no break; "
             "the in-frame send/throw form of C42 is triggered by a break.\n")
    open(os.path.join(a.out, f"P0-1_{a.backend}.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:14]))
    print(f"... written {os.path.join(a.out, f'P0-1_{a.backend}.md')}")


if __name__ == "__main__":
    main()
