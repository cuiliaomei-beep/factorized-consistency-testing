# Running the GPU campaign in a Kaggle Notebook

## One-time setup

1. Register at kaggle.com and bind a phone number under **Settings → Phone Verification** (without it there is no GPU and no network access).
2. Package locally: in `toxic_compile/`, run `powershell -File plan\kaggle\pack.ps1` to obtain `plan_kaggle.zip` (code + corpus + mined reproducers, without local results).
3. In the Kaggle left sidebar, **Datasets → New Dataset**, upload `plan_kaggle.zip` and `plan/kaggle/bootstrap.sh`, and set the title to `tcc-plan` (the URL will be `/kaggle/input/tcc-plan/`).

## Each run

1. **Code → New Notebook**, Settings on the right:
   - Accelerator: **GPU T4 x2** (or P100);
   - Internet: **On** (pip needs to install nnsmith, and deduplication needs access to GitHub);
   - Persistence: Files only (optional).
   - Add Input → select your `tcc-plan` dataset.
2. First cell:

   ```python
   import os, shutil, glob
   src = os.path.dirname(glob.glob("/kaggle/input/**/run.py", recursive=True)[0])
   shutil.copytree(src, "/kaggle/working/plan", dirs_exist_ok=True)
   %cd /kaggle/working/plan
   !pip -q install nnsmith certifi expecttest hypothesis
   !python run.py selfcheck --offline
   !python run.py selfcheck
   ```

   (Kaggle unpacks the uploaded zip automatically, so glob is used to find `run.py`; `expecttest`/`hypothesis` are dependencies of the OpInfo database, and without them the 4th campaign gets 0 programs.)

   It is fine once you see `cuda True Tesla T4`, `triton x.y.z`, `29/29 checks passed`.
3. Second cell runs the campaign (ordered by value; you may run only the first few steps):

   ```python
   !bash /kaggle/working/plan/kaggle/campaign_gpu.sh
   ```

   A single campaign takes 1–2 hours; the free quota is 30 GPU hours per week, and a single session lasts at most 12 hours.

   **Since 2026-09-10, step 0 of the script is the scan of the new angles** (the ones most productive on CPU, not yet run on GPU): `decomp` (decomposition/meta differential, no compilation, 10 minutes), `metamorphic` (metamorphic relations, including GPU-specific configurations cudagraphs / persistent_reductions / max_autotune), `binding` (binding forms), `aoti` (export + AOTInductor, Triton path), `diskcache`. To run only these:

   ```python
   !cd /kaggle/working/plan && python run.py decomp --isolate --samples 2 --out gpu_results_decomp
   !cd /kaggle/working/plan && python run.py metamorphic --isolate --samples 2 --seed 2 --device cuda --relations config --configs dynamic,max_autotune,cudagraphs,no_persistent_reductions --out gpu_results_meta_cfg
   !cd /kaggle/working/plan && python run.py aoti --isolate --samples 1 --device cuda --out gpu_results_aoti
   ```

   Results are in `gpu_results_*/` (`DECOMP_DIFF.md` / `METAMORPHIC.md` / `AOTI.md` / `BINDING.md`); after downloading, put them back into the local `plan/results/` and triage them with `scripts/decomp_triage.py` etc. Step 2 is real models (`reproducers_models/`, torchvision is preinstalled).
4. **To keep it running after you close the browser**: top right **Save Version → Save & Run All (Commit)**. When the committed run finishes, `/kaggle/working/plan/gpu_reports_*` can be downloaded from the Notebook's **Output** page (each directory contains `SUMMARY.md`, `CAMPAIGN.md`, and an `issue.md` for each candidate).
5. Put the downloaded `gpu_reports_*` back under the local `plan/` and process them with the local triage workflow (`TO_SUBMIT.md` (internal notes, not included), `campaign --status`).

## Isolated execution on GPU

A CUDA device-side assertion invalidates the CUDA context of the whole process, so on GPU `campaign` defaults to `--isolate`: each reproducer (or each batch of 25 generated programs) runs in its own subprocess; the parent process prints only scheduling lines such as `[i/N] --file xxx exit 0 [12s]` and the subprocess's candidate summary, and a crashed subprocess is recorded as `exit 1`/`timeout` before continuing.

## Notes

- An interactive session disconnects after 20 minutes of inactivity; always use Save & Run All for long tasks.
- Every step supports resuming from a checkpoint (`progress.json`); if a session is killed, at most one program is lost. When re-running Run All, keep the `--out` directory in Output and mount it back to resume.
- On GPU, Inductor generates Triton kernels, which is layer E4 in the plan; `--device cuda` moves all seed tensors to the GPU, and the cpu↔cuda switch of the `device` factor also takes effect automatically.
- The steps on Colab are the same (Runtime → Change runtime type → T4), except that `/kaggle/input/...` is replaced by a Google Drive path.
