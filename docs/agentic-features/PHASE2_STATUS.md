# Phase 2 Status — Real HF Jobs Training for Feature C

**Status:** Blocked on HF account credits. Pipeline ready; just needs funding.

## What's verified
- HF account `ander2221` can reach the Jobs API with the token at `~/.cache/huggingface/token`.
- Token fine-grained permissions include `job.write`.
- `hf jobs ps` returns OK; `hf jobs hardware` lists 19 flavors with pricing.
- `scripts/hf_jobs_smoke.py` runs cleanly end-to-end locally: clones the fork, pip-installs, runs all 10 VisDrone smoke tests, exits 0.

## What's blocked
Submitting any HF Jobs run (even `cpu-basic` at $0.0002/min) returns:

```
402 Payment Required
Pre-paid credit balance is insufficient - add more credits to your account to use Jobs.
```

The account is on the free tier (`isPro: false`) with no pre-paid balance. HF Jobs is pay-per-use; there is no free allocation.

## Cost to unblock Phase 2

| Artifact | Flavor | Wall-clock | Cost |
|----------|--------|-----------|------|
| Smoke sanity check | `cpu-basic` | ~3 min | **~$0.001** |
| VisDrone fine-tune (YOLOv9-s, 50 epochs, imgsz 640) | `a10g-small` | ~3 hours | **~$3** |
| Same but YOLOv9-m | `a10g-small` | ~5 hours | ~$5 |

So for ~$5 total credit, the full Phase 2 story can run: smoke sanity check + real VisDrone training + published HF Hub weights.

## How to unblock

1. Add a credit card to https://huggingface.co/settings/billing (one-time setup).
2. Pre-load some balance (HF bills usage from the balance — $10 leaves room for a few runs).
3. Re-run the smoke submission:
   ```bash
   hf jobs uv run scripts/hf_jobs_smoke.py \\
       --flavor cpu-basic --secrets HF_TOKEN --timeout 10m -d
   ```
4. Once the smoke returns 0, submit the real training:
   ```bash
   hf jobs uv run scripts/hf_jobs_visdrone_train.py \\
       --flavor a10g-small --secrets HF_TOKEN --timeout 6h -d
   ```

The second script isn't in the repo yet because there's no point building it without verified smoke on the same infra (dataset format handling is the main uncertainty — see below).

## Known follow-up: Voxel51 dataset format

The most accessible VisDrone mirror on the HF Hub is `Voxel51/VisDrone2019-DET`, but it's stored in FiftyOne format (flat `data/` directory plus a single `samples.json`), not the raw VisDrone layout the current converter expects. Two ways forward:

1. **Add a FiftyOne parser** to `scripts/finetune_yolo9_visdrone.py` — ~50 lines, one extra test case. Best long-term.
2. **Download from the official VisDrone URLs** (Google Drive hosted) using `gdown` inside the job. Already-documented fragility.

I'd pick (1) when Phase 2 unblocks — the smoke test covers the YOLO-side of the pipeline already; adding a FiftyOne adapter is pure data-plumbing.
