"""Bounded, CPU-capable driver for TorchProbe (soodoshll/temisu, commit 3aa039c).

The loop body is temisu/fuzz.py verbatim except:
  * device is a parameter (fuzz.py hard-codes 'cuda');
  * the `while True` loop is bounded (--seeds) and model_gen gets an explicit seed (fuzz.py passes none);
  * every test is additionally run *eagerly* (uncompiled) against the NNSmith oracle to tell an invalid
    mutation from a compiler divergence (fuzz.py attributes every failure to the compiler);
  * results go to a JSONL file; the mutated programs (code + mlist + inputs) are saved for Setting B.
"""
import argparse
import ast
import json
import logging
import os
import random
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "temisu"))
# Inductor's CPU backend on Windows needs the MSVC environment; run.py does the same at start-up.
PLAN = os.path.dirname(os.path.dirname(HERE))  # artifact root (contains tcc/)
sys.path.insert(0, PLAN)
from tcc.compat import ensure_msvc_env  # noqa: E402
print("MSVC:", ensure_msvc_env(verbose=True), flush=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch._dynamo  # noqa: E402
from nnsmith.materialize import Model  # noqa: E402
from nnsmith.graph_gen import model_gen  # noqa: E402
from nnsmith.narrow_spec import auto_opset  # noqa: E402
from nnsmith.util import op_filter  # noqa: E402
from temisu.ir import TFunction  # noqa: E402
from temisu.mutator import Mutator  # noqa: E402
from temisu.logging import TEMISU_LOG  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--first-seed", type=int, default=0)
ap.add_argument("--max-nodes", type=int, default=10)
ap.add_argument("--backend", default="inductor")
ap.add_argument("--out", default=os.path.join(HERE, "tp_out"))
args = ap.parse_args()
os.makedirs(os.path.join(args.out, "programs"), exist_ok=True)
device = args.device
TEMISU_LOG.setLevel(logging.WARNING)

# --- verbatim from fuzz.py (opset filter) ---
ModelType = Model.init("torch", backend_target=device)
ModelType.add_seed_setter()
opset = op_filter(
    auto_opset(ModelType, vulops=False),
    exclude=["core.LinearInterp", "core.BilinearInterp", "core.NearestInterp", "core.BicubicInterp",
             "core.TrilinearInterp",
             "core.Concat1", "core.Concat2", "core.Concat3", "core.Concat4", "core.Concat5", "core.Concat6",
             "core.Floor", "core.Round", "core.Ceil",
             "core.ExpandLast1", "core.ExpandLast2", "core.ExpandLast3", "core.ExpandLast4",
             ]
)


class Inconsistency(Exception):
    def __init__(self, annotation="", target=None, output=None):
        super().__init__(annotation)
        self.target = target
        self.output = output
        self.annotation = annotation


def _no_nan_or_inf(target):
    return not np.any(np.isinf(target)) and not np.any(np.isnan(target))


def verify_results(targets, outputs, tfunc, rtol=1e-05, atol=1e-08):
    if len(targets) != len(outputs):
        raise Inconsistency(f"len(targets) != len(outputs) ({len(targets)} vs {len(outputs)})")
    for i, var in enumerate(tfunc._model.output_map.keys()):
        target = targets[i]
        output = outputs[i]
        if not _no_nan_or_inf(target):
            continue
        if not np.allclose(target, output, rtol=rtol, atol=atol):
            raise Inconsistency(f"Inconsistent result {var}", target, output)


def _compile_and_run(prog, mlist, input_tensors, mode='default', backend='inductor'):
    with torch.no_grad():
        compiled = torch.compile(prog, backend=backend, mode=mode)
        output = compiled(mlist, **input_tensors)
    if not isinstance(output, tuple):
        output = (output, )
    return [o.detach().cpu().numpy() for o in output]


def _eager_run(prog, mlist, input_tensors):
    with torch.no_grad():
        output = prog(mlist, **input_tensors)
    if not isinstance(output, tuple):
        output = (output, )
    return [o.detach().cpu().numpy() for o in output]


def max_abs_diff(targets, outputs):
    m = 0.0
    for t, o in zip(targets, outputs):
        try:
            d = np.abs(np.asarray(t, dtype=np.float64) - np.asarray(o, dtype=np.float64))
            d = d[np.isfinite(d)]
            if d.size:
                m = max(m, float(d.max()))
        except Exception:
            return None
    return m


def clone_inputs(d):
    return {k: v.clone() for k, v in d.items()}


jsonl = open(os.path.join(args.out, "tests.jsonl"), "a", encoding="utf-8")


def emit(row):
    jsonl.write(json.dumps(row, default=str) + "\n")
    jsonl.flush()


t_all = time.perf_counter()
for s in range(args.first_seed, args.first_seed + args.seeds):
    random.seed(s)
    torch.manual_seed(s)
    np.random.seed(s)
    rec = {"seed": s}
    t0 = time.perf_counter()
    try:
        gen = model_gen(opset=opset, max_elem_per_tensor=65536, timeout_ms=10000, max_nodes=args.max_nodes,
                        dtype_choices=['bool', 'f32', 'int32', 'int64', 'f64', ], seed=s)
        ir = gen.make_concrete()
        model = ModelType.from_gir(ir)
        model.refine_weights()
        oracle = model.make_oracle()
    except Exception as e:  # noqa: BLE001
        rec.update({"status": "seed_generation_failed", "error": f"{type(e).__name__}: {str(e)[:200]}"})
        emit(rec)
        print(rec, flush=True)
        continue
    target = [oracle.output[k] for k in model.native_model.output_map]
    th_model = model.native_model
    input_tensors = {k: torch.tensor(v, device=device) for k, v in oracle.input.items()}
    tfunc = TFunction(th_model)
    th_model.to(device)
    mlist = th_model.mlist
    rec.update({"n_ops": ir.n_compute_inst(), "n_inputs": len(input_tensors), "n_outputs": len(target),
                "input_shapes": {k: list(v.shape) for k, v in input_tensors.items()},
                "input_dtypes": {k: str(v.dtype) for k, v in input_tensors.items()},
                "seed_wall_s": round(time.perf_counter() - t0, 2)})
    print(f"seed {s}: {rec['n_ops']} ops, inputs {rec['input_shapes']}", flush=True)
    for k, (mutation_name, tf) in enumerate(Mutator(tfunc, input_tensors)):
        row = dict(rec)
        row.update({"k": k, "mutation": mutation_name})
        t1 = time.perf_counter()
        if tf is None:
            row["status"] = "mutation_returned_none"
            emit(row)
            continue
        try:
            func = tf.fn()
            src = ast.unparse(tf._fn_ast())
        except Exception as e:  # noqa: BLE001
            row.update({"status": "render_failed", "error": f"{type(e).__name__}: {str(e)[:300]}"})
            emit(row)
            print("  ", row["mutation"], row["status"], row["error"][:100], flush=True)
            continue
        row["n_lines"] = src.count("\n") + 1
        pname = f"seed{s}_k{k}_{mutation_name}"
        with open(os.path.join(args.out, "programs", pname + ".py"), "w", encoding="utf-8") as fh:
            fh.write(src)
        try:
            torch.save((tf.patched_mlist(), clone_inputs(input_tensors)),
                       os.path.join(args.out, "programs", pname + ".pt"))
        except Exception as e:  # noqa: BLE001
            row["save_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        # extra: eager validity of the mutated program (not in fuzz.py)
        eout = None
        try:
            eout = _eager_run(func, mlist, clone_inputs(input_tensors))
            try:
                verify_results(target, eout, tf, 1e-2, 1e-4)
                row["eager_valid"] = True
            except Inconsistency as e:
                row["eager_valid"] = False
                row["eager_vs_oracle"] = e.annotation
                row["eager_maxdiff"] = max_abs_diff(target, eout)
        except Exception as e:  # noqa: BLE001
            row["eager_valid"] = False
            row["eager_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        # verbatim fuzz.py check: compiled vs NNSmith oracle
        torch._dynamo.reset()
        output = None
        try:
            output = _compile_and_run(func, mlist, clone_inputs(input_tensors), backend=args.backend)
            verify_results(target, output, tf, 1e-2, 1e-4)
            row["status"] = "pass"
        except Inconsistency as e:
            row.update({"status": "inconsistent", "error": e.annotation,
                        "maxdiff_compiled_vs_oracle": max_abs_diff(target, output)})
            if row.get("eager_valid") is True and eout is not None:
                row["compiled_vs_eager_maxdiff"] = max_abs_diff(eout, output)
        except Exception as e:  # noqa: BLE001
            row.update({"status": "compile_or_run_error", "error": f"{type(e).__name__}: {str(e)[:400]}",
                        "traceback_tail": traceback.format_exc()[-800:]})
        row["wall_s"] = round(time.perf_counter() - t1, 2)
        emit(row)
        print(f"   k{k} {mutation_name:<20} {row['status']:<22} eager_valid={row.get('eager_valid')} "
              f"{row['wall_s']}s " + (row.get("error", "")[:90] if row["status"] != "pass" else ""), flush=True)
print(f"total wall {time.perf_counter() - t_all:.1f}s", flush=True)
