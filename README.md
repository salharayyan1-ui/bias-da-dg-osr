# PA1 — Beyond IID: Inductive Biases, Domain Adaptation, Domain Generalization, Open-Set Recognition

EE-5102/CS-6304, Fall 2026 — Programming Assignment 1

Each task has one notebook (`taskN/taskN.ipynb`) that runs every required step in order and writes:

* machine-readable results to `taskN/results/` (every table as `.json` + `.csv`, training logs as `.metrics.jsonl`, run summaries, prediction arrays),
* figures to `report/figures/taskN_*.png` and `.pdf`.

Every number in the report should trace back to one of these files.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
```

Datasets go under `data/`, which git ignores. `python scripts/prepare_data.py` fetches everything up front:

| Task | Data | Primary source | Fallback (used automatically if the primary host is unreachable) |
|---|---|---|---|
| 1 | STL-10 (labelled train/test) | torchvision binaries, if already extracted in `data/stl10_binary/` | HuggingFace `tanganke/stl10` (same 5,000 / 8,000 images and class order) |
| 1 | AdaIN weights | GitHub release of naoto0804/pytorch-AdaIN → `task1/checkpoints/adain/` | — |
| 2, 3 | PACS | DomainBed's Google Drive archive via `gdown` → `data/pacs/` | HuggingFace `flwrlabs/pacs` (same 9,991 images), written into the same `<domain>/<class>/` layout |
| 4 | CIFAR-10 / CIFAR-100 | torchvision (Toronto) | HuggingFace `uoft-cs/cifar10` / `uoft-cs/cifar100`, cached as `.npz` |

The reported results were produced with the HuggingFace copies, because the primary hosts were unreachable or very slow from the GPU machine. To reproduce them, force those copies with `python scripts/prepare_data.py --hf`. The committed PACS split manifest refers to the file names this path writes.

## Running

Open a notebook and choose *Run All*, or run it headless:

```bash
python scripts/run_notebook.py task1
python scripts/run_notebook.py task2 task3   # Task 2 must finish before Task 3
python scripts/run_notebook.py task4
```

* Task 3 loads Task 2's Source-only checkpoint as its ERM baseline, and its final cell compares against Task 2's saved predictions. Run Task 2 first.
* Each notebook has a `QUICK_RUN` flag (a few steps on subsets, for checking the pipeline only; do not report those numbers) and a `RETRAIN` flag (by default, existing checkpoints in `taskN/checkpoints/` are reused).
* Rough GPU budget: Task 1 < 1 h; Task 2, eight runs of ≤ 30 epochs; Task 3, four runs (SAM costs about 2× ERM); Task 4, 100 + 100 + 50 + 100 epochs on CIFAR.

Training can also be launched from the command line: `python task2/train.py --all`, `python task3/train.py --all`, `python task4/train.py --method vanilla`. These call the same functions as the notebooks and write the same checkpoints.

## Repository layout

```
common/     seeding, repo paths, logging/table saving, shared metrics, plotting helpers
shared/     PACS data, split manifest, MMD, and the single training/evaluation engine for Tasks 2-3
            splits/pacs_sketch_seed6304.json   (written on the first run, reused by both tasks)
task1/      data/ (subset, interventions, AdaIN cue conflicts), models/, analysis/, task1.ipynb
task2/      methods/ (source_only, dan, dann, cdan), models/, evaluation/, train.py, evaluate_final.py, task2.ipynb
task3/      methods/ (erm, dan_dg, sam), evaluation/ (sharpness, source separability), selection/, train.py, evaluate_sketch.py, task3.ipynb
task4/      data/, models/, methods/ (vanilla, gcsc, proser, manifold_mixup, rpl), scores/, evaluation/, pipeline.py, task4.ipynb
report/     figures/
scripts/    prepare_data.py, run_notebook.py (--quick for pipeline checks), run_queue.sh (quick checks + full runs)
```

## Protocol and design decisions

Seed 6304 is used everywhere: splits, subsets, patch permutations, head training, probes, the sharpness batch, and data-loader generators. It is also reset before every training run, so every method starts from the same initialization and sampling order.

**Task 1.**
* Dataset: STL-10. The 96×96 images are resized directly to the common 224×224 image, with no centre crop.
* Heads: linear heads are trained on cached frozen features (AdamW 1e-3 / 1e-4, batch 64, ≤ 50 epochs, patience 5 on validation accuracy). Each head standardizes its input with the training split's per-dimension mean and standard deviation. This is an affine map, so the head is still linear in the unchanged representation (see "Changes after run v1").
* Additional colour intervention: a luma-preserving 180° hue rotation in YIQ space.
* Cue conflicts: AdaIN with α = 1 over 6 animal/vehicle pairs in both directions. Style images are a central 60% crop of test images outside the evaluation subset. The rejection rule uses pixel statistics only: flat output, style not transferred, or edge-map correlation with the content below 0.45. That threshold was set from a pilot batch inspected by eye without any model predictions. The first 20 accepted images per direction are kept, giving 240.
* Projection: t-SNE (cosine metric, perplexity 30).
* Supplementary sweeps: hue angle, style strength α, and patch-grid size.

**Tasks 2–3.**
* Pipeline: one shared pipeline, `shared/pacs_engine.py`, with BatchNorm running statistics frozen.
* Batches: 8 images per source domain, plus 24 target images for UDA.
* Optimizer and selection: AdamW 1e-4 / 1e-4, ≤ 30 epochs, patience 5 on mean source-validation macro-F1.
* Separability probes: features are standardized before the C = 1 logistic regression, so the comparison is not confounded by feature scale.
* Design studies: Task 2 varies DAN λ_MMD ∈ {0.1, 1, 10}. Task 3 varies DAN-DG λ_DG ∈ {0.1, 1, 10}, so target-aware and target-free alignment can be compared at matched strength. An optional SAM ρ study is behind a flag. Task 2 also runs the assignment's other study option as a supplement, DANN's maximum GRL strength α_max ∈ {0.25, 0.5, 1}. It was added after the main DANN run had been evaluated (see "Changes after run v1"); the main DANN row stays at α_max = 1.
* MMD estimator: the unbiased U-statistic MMD² is used (DAN and DAN-DG share it). With 8 images per domain, the biased estimator is dominated by its diagonal terms and drove DAN-DG (λ = 1) into an all-zero feature collapse.
* DANN/CDAN discriminator input: the discriminator sees the feature's direction at a fixed norm, √512·f/‖f‖ (CDAN: vec(√512·f/‖f‖ ⊗ p)). With frozen BN and AdamW, an un-normalized DANN diverged in its first epoch as the feature norm inflated; the diverged log is kept in `task2/results/diagnostics/`. The architecture, GRL schedule, loss weight and optimizer are unchanged.
* Label isolation: Sketch labels are read only after a configuration-lock cell. Task 3 training and diagnostics receive split dictionaries that do not contain Sketch at all.

**Task 4.**
* Unknowns: CIFAR-100 test images only, loaded after the lock cell.
* Threshold: the 95th percentile of each score on CIFAR-10 validation.
* PROSER: dummy logits are max-collapsed as in the paper. Its placeholder score is `max dummy − max known logit`; thresholding this at the validation 95th percentile is exactly the paper's bias calibration. The reference's P(dummy) score is reported as an extra row. Mixed (manifold-mixup) hidden states pass through layer3/layer4 without updating those BatchNorm layers' running statistics (momentum 0 for that pass), so running statistics come from real images only; the mixed batch is still normalized with its own statistics, as in the reference. The checkpoint is the best CIFAR-10 validation accuracy, as required; the final-epoch model is also reported, in rows marked `proser_last_*`.
* RPL (the optional extension) is enabled.

## Changes after run v1

The first full run (v1) was checked against the diagnostics the assignment asks for. Three methods had not trained as intended. Each fix below was motivated only by information the protocol allows at that stage: training curves, source or CIFAR-10 validation data, and (Task 2) domain identities. No target class label and no CIFAR-100 image was used to choose a fix. The rerun (v2) is reported whether it raised or lowered target and open-set numbers, and the v1 logs are kept under `taskN/results/diagnostics/v1_*`.

Because the fixed checkpoints could not be moved to the new GPU machine reliably, v2 retrains **every** model from scratch with the same code, seeds and data, in one end-to-end run on an RTX 3060. With deterministic cuDNN the pipeline reproduced across GPUs: every unchanged method selected the same epoch and produced exactly the v1 numbers, so only the fixed methods' results differ from v1. The data are bit-identical: the regenerated PACS manifest, the CIFAR-10 split and the STL-10 subset were checked against the committed files.

1. **Task 2, DANN/CDAN discriminator input (unit norm → fixed norm √512).** v1 fed the discriminator f/‖f‖. Its domain loss stayed near ln 2 (≈0.64 in the first epoch, GRL α < 0.17), and its accuracy then fell to chance, while a linear probe on the frozen features still separated source from Sketch at 99.6 %. `task2/diagnose_discriminator.py` isolates the input scale: frozen Source-only features, the prescribed discriminator and optimizer, no gradient reversal, no Sketch labels (`task2/results/diagnostics/discriminator_input_scale.json`). With unit-norm inputs the discriminator gets the sign of the domain right, but its logits stay near zero (much higher held-out cross-entropy than at √512). The normalization also divides the reversed gradient reaching the features by ‖f‖ ≈ 30–55. So the adversarial game was played at near-zero logits with a weak gradient into the backbone. √512 keeps the protection against norm inflation at the raw feature's per-coordinate scale.
   In v2, from the second epoch on, the backbone overpowers the discriminator: its training accuracy stays *below* chance (0.33–0.46) with its loss above ln 2, and the Sketch predictions collapse onto one class. Both signs are visible without target labels. The resulting DANN is reported as the main DANN result, as committed before the rerun. Because this failure mode depends on the adversarial strength, the assignment's other design study (α_max ∈ {0.25, 0.5, 1}) was added as a supplement. Its values are the assignment's, and its results are for analysis only.
2. **Task 4, PROSER BatchNorm handling during the mixup pass, plus a final-epoch row.** In v1, mixed hidden states (lower variance than real ones) also updated the layer3/layer4 running statistics. CIFAR-10 validation accuracy fell from 0.948 to 0.886 within four epochs, and the checkpoint rule kept epoch 0, before the data placeholders had been learned (data-placeholder loss still ≈ 3.0). An intermediate PROSER run (logged as the `v2_*` files in `task4/results/diagnostics/`) put those BatchNorm layers in eval mode for the mixed pass. That changes the training computation and made training erratic: validation accuracy swung between 0.72 and 0.94, and the losses rose over the last epochs while the learning rate was ≈ 1e-6. It was therefore replaced, on CIFAR-10 evidence alone, by the final version: the same train-mode computation as v1, with momentum 0 for the mixed pass so it no longer updates the running statistics. With identical training losses, epoch-0 validation accuracy is 0.950 instead of v1's 0.948. It then stays between 0.945 and 0.953 for all 50 epochs, and the checkpoint rule selects epoch 21. The main rows still use the best-validation checkpoint; the final-epoch model is an extra row, fixed before evaluation. The v1 and v2 logs and tables are in `task4/results/diagnostics/`.
3. **Task 1, standardized head inputs.** The v1 CLIP head, on a unit-norm embedding, stopped (early stopping on validation accuracy, which saturated at epoch 0) with logits about 20× too small: 98.4 % accuracy at 0.59 mean confidence and validation loss 0.35, versus 0.05–0.06 for ResNet-50 and ViT-B/16. Its confidences were therefore not comparable with the other models or with zero-shot CLIP. All three heads now standardize their input (see Task 1 above).

`task2/configs/dan.yaml` also listed `mmd_estimator` twice; it now states the estimator used (unbiased).

## Attribution

* Backbones: torchvision (ResNet-50 `IMAGENET1K_V2`, ViT-B/16 `IMAGENET1K_V1`, ResNet-18 `IMAGENET1K_V1`) and OpenCLIP (`ViT-B-32`, `pretrained="openai"`).
* AdaIN (Huang & Belongie, 2017): the encoder and decoder architectures and the pretrained weights `vgg_normalised.pth` / `decoder.pth` come from https://github.com/naoto0804/pytorch-AdaIN (release v0.0.0). `task1/data/make_cue_conflicts.py` reproduces its `net.py` layer definitions so the checkpoints load unchanged.
* PACS download: the Google Drive archive used by DomainBed (https://github.com/facebookresearch/DomainBed).
* SAM two-step update: follows https://github.com/davda54/sam.
* PROSER losses and placeholder score: follow Zhou et al. (2021) and https://github.com/zhoudw-zdw/CVPR21-Proser.
* RPL (Chen et al., 2020): distance and open-space loss follow the reference `RPLoss`/`Dist` in https://github.com/iCGY96/ARPL.

This code was written with LLM coding assistance, which the assignment's coding policy permits. The PDF report is written separately, without generative AI.
