# Confirming issue draft 07 (pdist zero-row backward crash) on Linux

There is no WSL on the local machine. Three installation-free options; pick any one.

## A. Kaggle (recommended, existing workflow)

Create a new Notebook (CPU is enough, no GPU needed) with two cells:

Cell 1 — install torch 2.14.0 CPU, the same version as in the issue (about 1–2 minutes):

```
!pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

Cell 2 — paste the script directly (does not depend on the dataset):

```python
import platform, signal, subprocess, sys, torch

CASE = r'''
import sys, torch
n, d = int(sys.argv[1]), int(sys.argv[2])
x = torch.randn(n, d, requires_grad=True)
y = torch.nn.functional.pdist(x)
print("forward ok, output shape", tuple(y.shape), flush=True)
y.sum().backward()
print("backward ok, grad shape", tuple(x.grad.shape), flush=True)
'''
print("torch", torch.__version__, "|", platform.platform(), "|", sys.version.split()[0])
for n, d in ((0, 1), (0, 4), (1, 4), (2, 4)):
    r = subprocess.run([sys.executable, "-c", CASE, str(n), str(d)], capture_output=True, text=True)
    out = " | ".join(r.stdout.strip().splitlines())
    if r.returncode == 0:
        status = ""
    elif r.returncode < 0:
        status = f"   <-- process killed by signal {-r.returncode} ({signal.Signals(-r.returncode).name})"
    else:
        status = f"   <-- process died, exit code {r.returncode & 0xFFFFFFFF:#x}"
    print(f"pdist on ({n}, {d}): {out}{status}")
```

Expected output (Linux): the first two lines show `killed by signal 8 (SIGFPE)`, the last two are normal. Paste this output together with the version information on the first line into the issue's
"Error logs"/description (`torch 2.14.0+cpu | Linux-...-x86_64 | 3.11.x`).

If you do not want to reinstall torch in Cell 1, running Cell 2 directly also works: Kaggle's bundled torch 2.10 crashes as well, which can serve as a "multi-version reproduction" supplement, but the main body of the issue is still based on 2.14.0.

## B. Google Colab

Same two cells as above, unchanged.

## C. GitHub Actions (leaves a public log link)

Add `.github/workflows/pdist.yml` to any repository of your own:

```yaml
name: pdist-crash
on: workflow_dispatch
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
      - run: |
          curl -sL <relative path after putting pdist_crash_linux.py into the repository> || true
          python pdist_crash_linux.py
```

Commit `plan/kaggle/verify/pdist_crash_linux.py` to the repository, trigger it manually from the Actions page, and the log URL can be pasted into the issue.

## It can also be filed without extra verification

The CPU kernel of `pdist_backward` has no platform-specific branch, and the division by zero happens in the same piece of C++; the issue draft (issue drafts in 0920issues/, internal notes, not included) already states "verified on Windows
(0xC0000094); expected SIGFPE on Linux", and maintainers usually confirm on Linux themselves. The three options above only make the report stronger.
