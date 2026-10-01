# Historical bug benchmark (plan §21)

`historical_bugs.json` is only a skeleton of the list: its fields correspond exactly to the ten fields of plan §21, and `entries` is empty,
because the local machine has only one PyTorch version (2.14.0+cpu), so the buggy and fixed versions cannot both be verified,
and unverified issue numbers should not be written into the ground truth.

Filling procedure:

1. From the toxic-compilation dataset, select events of the four categories code-generation deviation / JIT specialization /
   cache integrity / wrong generated code;
2. Check each one against the six criteria of §21 (Python reproducer, clear eager behavior, runnable compiled behavior,
   buggy version, fixed version, genuinely a consistency issue);
3. Write the reproducer as a `.py` file in this directory: define `f` (or a `torch.compile` target) and `args`;
4. Register an entry in `entries` and set `Verified` to `true`;
5. `python run.py bench --historical benchmark/historical_bugs.json` runs each entry on the currently installed PyTorch,
   and records the expected result according to "whether the current version is the BuggyVersion" (detected / should not be detected).

Until the historical benchmark is available, RQ1 uses the 19 injected faults in `tcc/faults.py` as controlled ground truth:
each fault corresponds to one class of problem in plan §1 (codegen deviation / specialization / cache),
with known triggering factors and the oracle that should fire; the "fixed version" is the real compiler (no injection).
