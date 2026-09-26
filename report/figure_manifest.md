# Figure manifest

Every figure is written by its task notebook to `report/figures/` as PNG and PDF. Figures marked **req** cover the assignment's "Required Evidence". The rest are supplementary, for the appendix or for picking from.

## Task 1 (`task1/task1.ipynb`)
| File | Content |
|---|---|
| `task1_interventions` | example images: clean / grayscale / hue +180° / shift / patch shuffle |
| `task1_head_training` | linear-head loss and validation accuracy |
| `task1_cue_conflict_contact_sheet` | accepted vs. rejected stylizations (made before model evaluation) |
| `task1_shape_texture_decisions` **req** | shape / texture / other fractions, with shape bias and coverage |
| `task1_cue_conflict_examples` **req** | agreements, disagreements and failures, with each model's predictions |
| `task1_style_strength_sweep` | shape bias and coverage vs. AdaIN α |
| `task1_hue_angle_sweep` | accuracy and consistency vs. hue rotation angle |
| `task1_translation` **req** | accuracy and consistency vs. displacement (mean ± std over directions) |
| `task1_patch_grid_sweep` | 2×2 / 4×4 / 8×8 shuffling |
| `task1_patch_shuffle_confidence`, `task1_patch_shuffle_failures` | confidence after shuffling and confident wrong predictions |
| `task1_representation_stability` **req** | normalized cosine stability per backbone and intervention |
| `task1_stability_vs_consistency` | prediction-level vs. representation-level change |
| `task1_tsne_grid` **req** | t-SNE of clean vs. transformed features, backbone × intervention |

## Task 2 (`task2/task2.ipynb`)
| File | Content |
|---|---|
| `task2_training_curves` **req** | class loss, source-val F1, MMD² or domain loss + discriminator accuracy + GRL α |
| `task2_separability_vs_accuracy` | domain separability vs. Sketch accuracy |
| `task2_per_class_delta` **req** | per-class Sketch accuracy change vs. Source-only |
| `task2_target_confusions` | row-normalized Sketch confusion matrices |
| `task2_failure_examples` **req** | Sketch images fixed or broken by adaptation |
| `task2_domain_tsne` | source-val and Sketch features, coloured by domain |
| `task2_dan_lambda_study` **req**, `task2_dan_lambda_training_curves` | λ_MMD study |
| `task2_dann_alpha_study`, `task2_dann_alpha_training_curves` | supplementary DANN α_max study (added after the main DANN run; analysis only) |

## Task 3 (`task3/task3.ipynb`)
| File | Content |
|---|---|
| `task3_training_curves` **req**, `task3_dan_dg_lambda_training_curves` | training curves (with pairwise MMD²) |
| `task3_sharpness_radii` | sharpness proxy vs. radius (0.05 is the required one) |
| `task3_per_class_delta` **req**, `task3_sketch_confusions` | per-class Sketch changes vs. ERM |
| `task3_vs_task2_per_class_delta` **req** | ERM / DAN (T2) / DAN-DG / SAM / DANN / CDAN per class |
| `task3_failure_examples` | fixed / broken / all-wrong Sketch images |
| `task3_dan_dg_lambda_study` **req** | λ_DG study |

## Task 4 (`task4/task4.ipynb`)
| File | Content |
|---|---|
| `task4_training_curves` | losses and CIFAR-10 validation accuracy |
| `task4_score_distributions_roc` **req** | MSP / MLS / Mahalanobis distributions (known / near / far, τ) and ROC |
| `task4_model_comparison` | CSA, AUROC and rejection for Vanilla / GCSC / PROSER / RPL |
| `task4_acceptance_heatmap` | accepted unknown classes × absorbing CIFAR-10 class |
| `task4_failure_examples` **req** | most confidently accepted near and far unknowns (class, prediction, score, τ) |
