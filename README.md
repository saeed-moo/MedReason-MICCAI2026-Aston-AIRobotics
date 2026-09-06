# MedReason 2026 — Aston-AIRobotics

A task-routed hybrid vision–language system for medical visual question answering (VQA), and a calibrated analysis of the open-ended reasoning ceiling.

This repository contains the code for our submission to the **MedReason 2026 Challenge** (MICCAI 2026 Satellite Events). The accompanying paper is accepted to the MICCAI 2026 Springer LNCS proceedings.

## Overview

The MedReason challenge combines two task types over medical images: closed-ended multiple-choice (MCQ) questions and open-ended questions, each requiring an image-grounded reasoning trace. Systems are scored on MCQ accuracy and two open-ended metrics (ground-truth agreement, GT; and visual accuracy, VA).

Our central finding is that **closed-ended and open-ended performance pull the backbone in opposite directions.** LoRA fine-tuning raises MCQ accuracy from 0.33 to 0.945, but *shortens* open-ended reasoning traces (44.3 → 19.9 words) and lowers open-ended quality. Rather than compromise, we route each question type to the configuration that serves it best.

## Method: task-routed hybrid

A single **Qwen2.5-VL-7B** backbone carries one LoRA adapter, toggled per question type:

- **MCQ** → adapter **enabled** (fine-tuned accuracy 0.945)
- **Open-ended** → adapter **disabled** (base model's richer, image-grounded traces)

This gives the fine-tuned model's MCQ accuracy and the base model's open-ended behaviour from one checkpoint plus a ~40MB adapter, rather than two full models. All inference is single-pass, deterministic, and self-contained (no external network).

## Key results (held-out validation)

| Configuration | MCQ acc | Open F1 | Trace words |
|---|---|---|---|
| Base Qwen2.5-VL | 0.33 | 0.171 | 44.3 |
| Fine-tuned (LoRA) | 0.945 | 0.166 | 19.9 |
| **Hybrid (submitted)** | **0.945** | **0.171** | **44.3** |

## Negative results (reported transparently)

Four natural interventions that did **not** improve open-ended quality under leakage-controlled evaluation:

- **MCQ self-consistency** — vote@5 = greedy = 0.945 (Δ +0.000), 6× cost.
- **DINOv3 retrieval augmentation** — an apparent gain (F1 0.285) was traced to validation images retrieving themselves (self-leakage, top-1 similarity 1.000). A leakage-free train-only index scored 0.159, *below* baseline.
- **Answer-style prompt engineering** — no variant beat the existing prompt.
- **Open-ended-only LoRA** — appeared marginally better on token F1 (0.175 vs 0.171), but the local judge reversed this: 0.812 vs 1.084 GT, substantially worse.

## Calibrated local judge

Token-overlap F1 and embedding cosine cannot resolve open-ended quality (four systems within 0.016 F1). We built a local LLM judge (Qwen2.5-32B-Instruct, 0–4 rubric) approximately anchored to the organizers' reported baseline. It reverses a spurious token-F1 "win" and, via error analysis, points to the 7B backbone's **visual-diagnostic perception** as the plausible bottleneck: the model confidently names incorrect specific findings on subtle/rare cases.

## Repository structure

```
src/
  training/      LoRA fine-tuning, adapter merging
  inference/     hybrid system, container test
  evaluation/    scoring, calibrated judge, error analysis, ablations
  data_prep/     dataset construction
slurm/           HPC job scripts
docker_submission/  custom_system.py, Dockerfile, requirements.txt
```

## Notes

- Model weights and challenge data are **not** included (challenge data-use terms; size). Paths point to the released MedReason training data and the public Qwen2.5-VL checkpoint.
- Compute: single NVIDIA GH200 GPU.

  ## Paper

This work is described in our paper accepted at the **MedReason Challenge & Workshop, MICCAI 2026** (Springer LNCS proceedings, poster):

> Saeed Moosivand, Fangyijie Wang, Ziyang Wang. *A Task-Routed Hybrid Vision–Language System for Medical Visual Question Answering, and a Calibrated Analysis of the Open-Ended Reasoning Ceiling.* MICCAI 2026 MedReason Workshop.

Full proceedings reference to be added once published.

## Citation

If you find this useful, please cite the accompanying paper (MICCAI 2026 MedReason workshop, Springer LNCS). Full reference to be added once proceedings are published.
