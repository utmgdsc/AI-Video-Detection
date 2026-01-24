# Dataset Evaluation Guide

## How to Evaluate Dataset Candidates

When researching potential datasets for deepfake/manipulated video detection, document each candidate using the table format in `candidates.md`.

## Fields to Document

| Field | Description |
|-------|-------------|
| **Name** | Dataset name |
| **Link** | URL to dataset page or paper |
| **License/Terms** | Usage restrictions, academic-only, commercial OK, etc. |
| **Size** | Number of videos/images, total GB if known |
| **Labels** | What labels are provided (real/fake, manipulation type, etc.) |
| **Modality** | Video, frames, audio, or combination |
| **Split Availability** | Does it provide train/val/test splits? |
| **Preprocessing Needed** | Face extraction, resizing, frame sampling, etc. |
| **Known Issues** | Quality problems, label noise, download difficulty |
| **Why It Fits** | Relevance to our project goals |

## Evaluation Criteria

When comparing candidates, prioritize:

1. **License compatibility** — Can we use it for academic research?
2. **Label quality** — Are labels reliable and well-documented?
3. **Size** — Large enough for meaningful experiments, small enough to handle
4. **Relevance** — Does it cover the manipulation types we want to detect?
5. **Accessibility** — Can we actually download and use it?

## Files in This Folder

- `candidates.md` — Table of all datasets considered
- `selected.md` — Documentation for the chosen dataset
