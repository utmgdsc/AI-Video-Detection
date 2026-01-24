# AI-Video-Detection (Deepfake / Manipulated Video Detection)

## Motivation

Deepfakes and AI-generated manipulated videos pose significant threats to information integrity, personal privacy, and public trust. As generative AI techniques become more sophisticated and accessible, the ability to create convincing fake videos has outpaced detection capabilities. This project aims to investigate and compare state-of-the-art detection methods to contribute to the ongoing effort to identify manipulated media.

## Problem Statement

Given a video (or extracted frames), determine whether the content has been artificially manipulated or generated using AI techniques such as face-swapping, lip-syncing, or full synthetic generation. The challenge lies in:
- Detecting subtle artifacts that distinguish real from fake
- Generalizing across different manipulation methods
- Handling compressed or degraded video quality

## Goals / Deliverables

| Deliverable | Description |
|-------------|-------------|
| Dataset selection + justification | Documented rationale for chosen dataset with license verification |
| Model investigation | 4 models researched: paper summaries, repo analysis, pretrained weight availability |
| Consistent documentation | Each model has standardized notes and setup guides |
| Experiments + results comparison | Train/evaluate each model; results table with metrics |
| Final report + slides/presentation | Written report summarizing findings; presentation slides |
| Peer evaluation | End-of-project peer evaluation |

## Approach Overview

1. **Dataset selection**: Evaluate candidate datasets (FaceForensics++, Celeb-DF, DFDC, etc.) based on size, license, label quality, and relevance
2. **Model investigation**: Each team member investigates one model:
   - XceptionNet
   - EfficientNet-based detectors
   - MesoNet
   - AASIST (audio-based, for multimodal comparison)
3. **Baseline experiments**: Run each model on the selected dataset with consistent preprocessing
4. **Documentation**: Record setup issues, fixes, and learnings in standardized templates
5. **Comparison**: Compile results into a comparison table

## Evaluation Plan

- **Data split**: Use standard train/val/test splits from dataset (or 70/15/15 if not provided)
- **Metrics**:
  - Accuracy
  - F1 Score
  - AUC-ROC (where applicable)
  - Per-class precision/recall
- **Consistency**: All models evaluated on the same test set with the same preprocessing

## Risks + Mitigations

| Risk | Mitigation |
|------|------------|
| Dataset license restrictions | Verify license before use; document terms in dataset selection |
| Compute constraints | Use pretrained weights where available; limit training epochs |
| Reproducibility issues | Document exact environment, versions, and commands |
| Model repo not maintained | Fork repo; document any fixes applied |
| Time constraints | Prioritize documentation over perfect results |